#include<cstdlib>
#include<ctime>

#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"
#include "utils.hpp"

#define WPT 8 // Hyperparameter to maximize register spilling of local memory
#define STEPS 12000000

using SYSTEMX::core::Driver;

__global__ void alu_compute_kernel(float *d, const float seed) {  
  float tmps[WPT];
  int id = blockDim.x * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < WPT; i++) {
    tmps[i] = id / 3.0f; // some random initialization to avoid gmem access
#pragma unroll
    for (int j = 0; j < STEPS; j++) {
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
      d[id * WPT + j] = sum;
    }    
  }
}

void Driver::aluComputeRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  srand((unsigned int)time(NULL));
  const int seed = rand();

  float *d_in;
  CUDA_CALL(cudaMallocAsync(&d_in, args->dimGrid.x * args->dimBlock.x * sizeof(float), args->stream)); 

  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  alu_compute_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (d_in, seed);
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);
  double gflops = 2.0 * (STEPS * WPT + WPT) * total_threads / elapsed_ms * 1e3 / 1e9;
  spdlog::info("{}(id: {}) {:.2f} Gflops {:d} ms", FUNC_NAME(alu_compute_kernel), args->id, gflops, elapsed_ms);

  // cleanup
  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}