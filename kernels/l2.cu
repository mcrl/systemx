#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"

#define L2_LOAD_STEPS 60

using SYSTEMX::core::Driver;

__global__ void l2_load_kernel(float *in, const int in_size, const int stride) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  
  // To avoid kernel optimization
  float sum = int2floatCast(0);
  for (int i = 0; i < L2_LOAD_STEPS; i++) {
    int idx = id;
    
    for (int j = 0; j < in_size / stride; j++) {
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
      idx = (idx + stride) % in_size;
    }
  }
  
  // fake store to `in` to avoid compiler optimization
  if (sum == int2floatCast(-1)) {
    in[id] = sum;
  }
}

void Driver::l2LoadRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  const int l2CacheSize = device_properties_.l2CacheSize;
  const int stride = device_properties_.warpSize; 
  const int in_size = l2CacheSize / sizeof(float); // in_size is set to L2 cache size, which will make
                                                  // L2 hit rate ~100%
  const int intra_step_access_per_thread = in_size / stride;
  
  float *d_in;
  CUDA_CALL(cudaMallocAsync(&d_in, in_size * sizeof(float), args->stream));
  CUDA_CALL(cudaMemsetAsync(d_in, 0.0f, in_size * sizeof(float), args->stream));
  
  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  l2_load_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (d_in, in_size, stride);
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);

  double per_thread_bandwidth = L2_LOAD_STEPS * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s {:d} ms", FUNC_NAME(l2_load_kernel), args->id, bandwidth, (int)elapsed_ms);
  
  // cleanup
  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}