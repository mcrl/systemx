#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"

using SYSTEMX::core::Driver;

// TODO: Consider different architectures
// Volta L2 cache is an 16-way set-associative cache with 6144 KiB capacity,
// a cache line of 64 B.
__global__ void l2_load_kernel(float *in, const int in_size, const int l2_cache_size,
                               const int stride, const int steps) {
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  
  // a register to avoid compiler optimization
  float tmp;

  for (int i = 0; i < steps; i++) {
    // each warp loads all data in l2 cache
    for (int j = 0; j < l2_cache_size; j += stride) {
      uint32_t block_offset = (bid * blockDim.x + j) % in_size;
      
      for (int k = 0; k < stride; k++) {
        uint32_t thread_offset = (tid + k) % stride; // since stride is warp size, this enables coalesced access
        // load from gmem bypassing l1 cache
        asm volatile(
          "{\n\t"
          "ld.global.cg.f32 %0, [%1];\n\t"
          "}"
          : "=f"(tmp)
          : "l"(&in[block_offset + thread_offset])
          : "memory");
      }
    }
  }

  // fake store to `in` to avoid compiler optimization
  if (tmp == int2floatCast(-1)) {
    in[bid * blockDim.x + tid] = tmp;
  }
}

void Driver::l2LoadRun() {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = createStream();

  const int maxThreadsPerBlock = device_properties_.maxThreadsPerBlock;
  const int maxThreadsPerMultiProcessor = device_properties_.maxThreadsPerMultiProcessor;
  const int multiProcessorCount = device_properties_.multiProcessorCount;
  const int l2CacheSize = device_properties_.l2CacheSize;

  const int stride = device_properties_.warpSize; 
  const int in_size = l2CacheSize; // To maximize L2 cache hit rate
  const int steps = 1; // Hyperparameter to set execution time 300ms

  float *in = (float *)Driver::mallocDBuf(in_size * sizeof(float));
  Driver::setDBuf(in, 0.0f, in_size * sizeof(float));
  spdlog::debug("in_size: {} l2CacheSize: {} stride {} steps {}", in_size, l2CacheSize, stride, steps);
  
  // Fully occupy half of total SMs
  dim3 gridDim((maxThreadsPerMultiProcessor / maxThreadsPerBlock) * (multiProcessorCount / 2), 1, 1);
  dim3 blockDim(maxThreadsPerBlock, 1, 1);
  l2_load_kernel << <gridDim, blockDim, 0, stream >> > (in, in_size, l2CacheSize, stride, steps);
}