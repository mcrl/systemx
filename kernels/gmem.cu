#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"

using SYSTEMX::core::Driver;

__global__ void gmem_load_kernel(float *in, const int in_size,
                                 const int stride, const int steps) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  
  // To avoid kernel optimization
  float sum = int2floatCast(0);
  for (int i = 0; i < steps; i++) {
    int idx = id;
    
    for (; idx < in_size;) {
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
  }
  
  // fake store to `in` to avoid compiler optimization
  if (sum == int2floatCast(-1)) {
    in[id] = sum;
  }
}

__global__ void gmem_store_kernel(float *out,
                                  const int out_size, const int stride,
                                  const int steps) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // To avoid kernel optimization
  register float src = int2floatCast(0);
  for (int i = 0; i < steps; i++) {
    int idx = id;
    
    for (; idx < out_size;) {
      register float *out_ptr = &out[idx];
      // store to gmem bypassing l1 cache
      asm volatile(
        "{\n\t"
        "st.global.cg.f32 [%0], %1;\n\t"
        "}"
        : "+l"(out_ptr)
        : "f"(src)
        : "memory");
      idx += stride;
    }
  }
}

void Driver::gmemLoadRun() {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = createStream();

  const int maxThreadsPerBlock = device_properties_.maxThreadsPerBlock;
  const int maxThreadsPerMultiProcessor = device_properties_.maxThreadsPerMultiProcessor;
  const int multiProcessorCount = device_properties_.multiProcessorCount;
  const int l2CacheSize = device_properties_.l2CacheSize;

  const int stride = l2CacheSize / sizeof(float);
  const int in_size = stride * 1024;
  const int steps = 117; // Hyperparameter to set execution time 100ms

  float *in = (float *)Driver::mallocDBuf(in_size * sizeof(float));
  
  // Fully occupy half of total SMs
  dim3 gridDim((maxThreadsPerMultiProcessor / maxThreadsPerBlock) * (multiProcessorCount / 2), 1, 1);
  dim3 blockDim(maxThreadsPerBlock, 1, 1);
  gmem_load_kernel << <gridDim, blockDim, 0, stream >> > (in, in_size, stride, steps);
}

void Driver::gmemStoreRun() {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = createStream();

  const int maxThreadsPerBlock = device_properties_.maxThreadsPerBlock;
  const int maxThreadsPerMultiProcessor = device_properties_.maxThreadsPerMultiProcessor;
  const int multiProcessorCount = device_properties_.multiProcessorCount;
  const int l2CacheSize = device_properties_.l2CacheSize;

  const int stride = l2CacheSize / sizeof(float);
  const int out_size = stride * 1024;
  const int steps = 107; // Hyperparameter to set execution time 100ms

  float *out = (float *)Driver::mallocDBuf(out_size * sizeof(float));

  // Fully occupy half of total SMs
  dim3 gridDim((maxThreadsPerMultiProcessor / maxThreadsPerBlock) * (multiProcessorCount / 2), 1, 1);
  dim3 blockDim(maxThreadsPerBlock, 1, 1);
  gmem_store_kernel << <gridDim, blockDim, 0, stream >> > (out, out_size, stride, steps);
}