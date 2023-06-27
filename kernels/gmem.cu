#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"

using SYSTEMX::core::Driver;

__global__ void gmem_load_kernel(float *in, float *out, const int stride, const int steps) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = id;
  
  // To avoid kernel optimization
  float sum = int2floatCast(0);
  for (int i = 0; i < steps; i++) {
    register float tmp;
    // load from gmem bypassing l1 cache
    asm volatile(
      "{\n\t"
      "ld.global.cg.f32 %0, [%1];\n\t"
      "}"
      : "=f"(tmp)
      : "l"(&in[idx])
      : "memory");
    sum += tmp;
    idx += stride;
  }
  
  // TODO: fake store to `out` to avoid compiler optimization
  if (sum == int2floatCast(-1)) {
    out[id] = sum;
  }
}

__global__ void gmem_store_kernel(float *in, float *out) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
}

void Driver::gmemLoadRun() {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = createStream();

  const int maxThreadsPerBlock = device_properties_.maxThreadsPerBlock;
  const int maxThreadsPerMultiProcessor = device_properties_.maxThreadsPerMultiProcessor;
  const int multiProcessorCount = device_properties_.multiProcessorCount;
  const int l2CacheSize = device_properties_.l2CacheSize;

  const int stride = l2CacheSize / sizeof(float);
  const int steps = 2;
  // TODO: Remove steps from input size. 
  // Input stride iterations do not depend on steps, as it can iterate over the same array over and over
  float *in = (float *)Driver::mallocDBuf(stride * steps * sizeof(float));
  float *out = (float *)Driver::mallocDBuf((maxThreadsPerMultiProcessor / maxThreadsPerBlock) * (multiProcessorCount / 2) *
                                           maxThreadsPerBlock * sizeof(float));

  spdlog::debug("l2CacheSize: {} stride {} steps {}", l2CacheSize, stride, steps);
  
  // Fully occupy half of total SMs
  dim3 gridDim((maxThreadsPerMultiProcessor / maxThreadsPerBlock) * (multiProcessorCount / 2), 1, 1);
  dim3 blockDim(maxThreadsPerBlock, 1, 1);
  gmem_load_kernel << <gridDim, blockDim, 0, stream >> > (in, out, stride, steps);
}

void Driver::gmemStoreRun() {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = createStream();

  const int maxThreadsPerBlock = device_properties_.maxThreadsPerBlock;
  const int maxThreadsPerMultiProcessor = device_properties_.maxThreadsPerMultiProcessor;
  const int multiProcessorCount = device_properties_.multiProcessorCount;
  
  // Fully occupy half of total SMs
  dim3 gridDim((maxThreadsPerMultiProcessor / maxThreadsPerBlock) * (multiProcessorCount / 2), 1, 1);
  dim3 blockDim(maxThreadsPerBlock, 1, 1);
}