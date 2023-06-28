#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"

using SYSTEMX::core::Driver;

__global__ void l2_load_kernel(float *in, const int in_size,
                               const int stride, const int steps) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void l2_store_kernel(float *out,
                                  const int out_size, const int stride,
                                  const int steps) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
}

void Driver::l2LoadRun() {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = createStream();

  const int maxThreadsPerBlock = device_properties_.maxThreadsPerBlock;
  const int maxThreadsPerMultiProcessor = device_properties_.maxThreadsPerMultiProcessor;
  const int multiProcessorCount = device_properties_.multiProcessorCount;
  const int l2CacheSize = device_properties_.l2CacheSize;

  const int stride = l2CacheSize / sizeof(float);
  const int in_size = stride * 1024;
  const int steps = NULL; // Hyperparameter to set execution time 100ms

  float *in = (float *)Driver::mallocDBuf(in_size * sizeof(float));

  spdlog::debug("in_size: {} l2CacheSize: {} stride {} steps {}", in_size, l2CacheSize, stride, steps);
  
  // Fully occupy half of total SMs
  dim3 gridDim((maxThreadsPerMultiProcessor / maxThreadsPerBlock) * (multiProcessorCount / 2), 1, 1);
  dim3 blockDim(maxThreadsPerBlock, 1, 1);
  l2_load_kernel << <gridDim, blockDim, 0, stream >> > (in, in_size, stride, steps);
}

void Driver::l2StoreRun() {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = createStream();

  const int maxThreadsPerBlock = device_properties_.maxThreadsPerBlock;
  const int maxThreadsPerMultiProcessor = device_properties_.maxThreadsPerMultiProcessor;
  const int multiProcessorCount = device_properties_.multiProcessorCount;
  const int l2CacheSize = device_properties_.l2CacheSize;

  const int stride = l2CacheSize / sizeof(float);
  const int out_size = stride * 1024;
  const int steps = NULL; // Hyperparameter to set execution time 100ms

  float *out = (float *)Driver::mallocDBuf(out_size * sizeof(float));

  spdlog::debug("out_size: {} l2CacheSize: {} stride {} steps {}", out_size, l2CacheSize, stride, steps);

  // Fully occupy half of total SMs
  dim3 gridDim((maxThreadsPerMultiProcessor / maxThreadsPerBlock) * (multiProcessorCount / 2), 1, 1);
  dim3 blockDim(maxThreadsPerBlock, 1, 1);
  l2_store_kernel << <gridDim, blockDim, 0, stream >> > (out, out_size, stride, steps);
}