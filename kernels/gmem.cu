#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"

using SYSTEMX::core::Driver;

__global__ void gmem_load_kernel(float *in, const int in_size,
                                 const int stride, const uint steps) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  
  // To avoid kernel optimization
  float sum = int2floatCast(0);
  for (uint i = 0; i < steps; i++) {
    int idx = id;
    
    for (uint j = 0; j < in_size / stride; j++) {
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

__global__ void gmem_store_kernel(float *out, const int out_size,
                                  const int stride, const uint steps) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // To avoid kernel optimization
  register float src = int2floatCast(0);
  for (uint i = 0; i < steps; i++) {
    int idx = id;
    
    for (uint j = 0; j < out_size / stride; j++) {
      register float *out_ptr = &out[idx];
      // store to gmem bypassing l1 cache
      asm volatile(
        "{\n\t"
        "st.global.cg.f32 [%0], %1;\n\t"
        "}"
        : "+l"(out_ptr)
        : "f"(src)
        : "memory");
      idx = (idx + stride) % out_size;
    }
  }
}

void Driver::gmemLoadRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  assertDeviceCorrect();

  const int l2CacheSize = device_properties_.l2CacheSize;
  const int stride = l2CacheSize / sizeof(float);
  const int intra_step_access_per_thread = 1024;
  const int in_size = stride * intra_step_access_per_thread;

  float *d_in;
  CUDA_CALL(cudaMallocAsync(&d_in, in_size * sizeof(float), args->stream));

  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);
  
  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  gmem_load_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (d_in, in_size, stride, args->steps);
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);

  double per_thread_bandwidth = args->steps * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s {:d} ms", FUNC_NAME(gmem_load_kernel), args->id, bandwidth, (int)elapsed_ms);

  // cleanup
  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}

void Driver::gmemStoreRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  assertDeviceCorrect();

  const int l2CacheSize = device_properties_.l2CacheSize;
  const int stride = l2CacheSize / sizeof(float);
  const int intra_step_access_per_thread = 1024;
  const int out_size = stride * intra_step_access_per_thread;

  float *d_out;
  CUDA_CALL(cudaMallocAsync(&d_out, out_size * sizeof(float), args->stream));

  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);
  
  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  gmem_store_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (d_out, out_size, stride, args->steps);
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);

  double per_thread_bandwidth = args->steps * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s {:d} ms", FUNC_NAME(gmem_store_kernel), args->id, bandwidth, (int)elapsed_ms);

  // cleanup
  CUDA_CALL(cudaFree(d_out));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}