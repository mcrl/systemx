#include<cstdlib>
#include<ctime>

#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"

#define WPT 8 // Hyperparameter to maximize register spilling of local memory
#define STEPS 100 * 1024 * 8 // Execution time is 100ms

using SYSTEMX::core::Driver;

inline __device__ float mad(const float a, const float b, const float c) {
  return a * b + c;
}

inline __device__ float int2floatCast(const int i) {
  return static_cast<float>(i);
}

__global__ void register_compute_kernel(float *d, float seed, int steps) {  
  float tmps[WPT];
  int id = WPT * blockDim.x * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < WPT; i++) {
    tmps[i] = id / 3.0f; // some random initialization to avoid gmem access
#pragma unroll
    for (int j = 0; j < steps; j++) {
      tmps[i] = mad(tmps[i], tmps[i], seed);
    }
  }

  // To avoid kernel optimization
  float sum = int2floatCast(0);
#pragma unroll
  for (int j = 0; j < WPT; j += 2) {
    sum = mad(tmps[j], tmps[j + 1], sum);
    // Never executed, to avoid kernel optimization
    // If not kernel execution is skipped
    if (sum == int2floatCast(-1)) {
      d[id + j] = sum;
    }    
  }
}

void Driver::registerComputeRun() {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = createStream();

  const int maxThreadsPerBlock = device_properties_.maxThreadsPerBlock;
  const int maxThreadsPerMultiProcessor = device_properties_.maxThreadsPerMultiProcessor;
  const int multiProcessorCount = device_properties_.multiProcessorCount;
  
  // Fully occupy half of total SMs
  dim3 gridDim(maxThreadsPerMultiProcessor / maxThreadsPerBlock, multiProcessorCount / 2, 1);
  dim3 blockDim(maxThreadsPerBlock, 1, 1);

  srand((unsigned int)time(NULL));
  const int seed = rand();
  float *d = (float *)Driver::mallocDBuf(sizeof(float) * WPT * maxThreadsPerBlock *
                               (maxThreadsPerMultiProcessor / maxThreadsPerBlock) *
                               (multiProcessorCount / 2));
  register_compute_kernel << <gridDim, blockDim, 0, stream >> > (d, seed, STEPS);
}