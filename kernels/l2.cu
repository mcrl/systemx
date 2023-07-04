#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"

#define STEPS 15

using SYSTEMX::core::Driver;

// TODO: Consider different architectures
// Volta L2 cache is an 16-way set-associative cache with 6144 KiB capacity,
// a cache line of 64 B.
__global__ void l2_load_kernel(float *in, const int in_size, const int l2_cache_size,
                               const int stride) {
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  
  // a register to avoid compiler optimization
  float tmp;

  for (int i = 0; i < STEPS; i++) {
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

void Driver::l2LoadRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  const int l2CacheSize = device_properties_.l2CacheSize;
  const int stride = device_properties_.warpSize; 
  const int in_size = l2CacheSize / sizeof(float); // To maximize L2 cache hit rate
  const int intra_step_access_per_thread = l2CacheSize / sizeof(float);
  
  float *d_in;
  CUDA_CALL(cudaMallocAsync(&d_in, in_size * sizeof(float), args->stream));
  CUDA_CALL(cudaMemsetAsync(d_in, 0.0f, in_size * sizeof(float), args->stream));
  
  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  l2_load_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (d_in, in_size, l2CacheSize, stride);
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);

  double per_thread_bandwidth = STEPS * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s", FUNC_NAME(l2_load_kernel), args->id, bandwidth);

  // cleanup
  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}