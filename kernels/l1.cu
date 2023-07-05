#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"

#define L1_LOAD_STEPS 14000
#define L1_STORE_STEPS 20000

using SYSTEMX::core::Driver;

// For Volta architecture, L1 cache and shared memory are combined with merged capacity of 128 KB.
// For example, if shared memory is configured to 64 KB, texture and load/store operations can use 
// the remaining 64 KB of L1. The shared memory is configurable up to 96 KB.

__global__ void l1_load_kernel(float *in, const int in_size, const int stride) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int nblocks = gridDim.x;
  int sector_size = in_size / nblocks;
  
  // To avoid kernel optimization
  register float sum = int2floatCast(0);
  for (int i = 0; i < L1_LOAD_STEPS; i++) {
    // Each block accesses to only its corresponding sector of out
    int sector_start = blockIdx.x * sector_size;
    int offset = threadIdx.x % sector_size;
    
    for (int j = 0; j < sector_size / stride; j++) {
      float *in_ptr = &in[sector_start + offset];
      sum += *in_ptr;
      offset = (offset + stride) % sector_size;
    }
  }

  // fake store to `in` to avoid compiler optimization
  if (sum == int2floatCast(-1)) {
    in[id] = sum;
  }
}

// TODO: This kernel is not complete yet.
__global__ void l1_store_kernel(float *out, const int out_size, const int stride) {
  int nblocks = gridDim.x;
  int sector_size = out_size / nblocks;
  
  // To avoid kernel optimization
  register float src = int2floatCast(0);
  for (int i = 0; i < L1_STORE_STEPS; i++) {
    // Each block accesses to only its corresponding sector of out
    int sector_start = blockIdx.x * sector_size;
    int offset = threadIdx.x % sector_size;
    
    for (int j = 0; j < sector_size / stride; j++) {
      float *out_ptr = &out[sector_start + offset];
      *out_ptr = src;
      offset = (offset + stride) % sector_size;
    }
  }
}

void Driver::l1LoadRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  // TODO: Remove hard-coded L1 cache + shared memory size in bytes
  // 128 KB for Volta, 192 KB for Ampere
  const int l1CacheSizeBytes = 128 * 1024 - device_properties_.sharedMemPerMultiprocessor;
  const int stride = device_properties_.warpSize;
  // in_size is set to L1 cache size * num blocks, which will make each block access to its
  // corresponding sector of d_in (which is of L1 cache size). This will make L1 hit rate ~100%.
  const int in_size = l1CacheSizeBytes * get_nblocks(args->dimGrid) / sizeof(float);
  const int intra_step_access_per_thread = l1CacheSizeBytes / sizeof(float) / stride;
  
  float *d_in;
  CUDA_CALL(cudaMallocAsync(&d_in, in_size * sizeof(float), args->stream));
  CUDA_CALL(cudaMemsetAsync(d_in, 0.0f, in_size * sizeof(float), args->stream));
  
  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  l1_load_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (d_in, in_size, stride);
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);

  double per_thread_bandwidth = L1_LOAD_STEPS * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s {:d} ms", FUNC_NAME(l1_load_kernel), args->id, bandwidth, (int)elapsed_ms);
  
  // cleanup
  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}

void Driver::l1StoreRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  // TODO: Remove hard-coded L1 cache + shared memory size in bytes
  // 128 KB for Volta, 192 KB for Ampere
  const int l1CacheSizeBytes = 128 * 1024 - device_properties_.sharedMemPerMultiprocessor;
  const int stride = device_properties_.warpSize;
  // out_size is set to L1 cache size * num blocks, which will make each block access to its
  // corresponding sector of d_out (which is of L1 cache size). This will make L1 hit rate ~100%.
  const int out_size = l1CacheSizeBytes * get_nblocks(args->dimGrid) / sizeof(float);
  const int intra_step_access_per_thread = l1CacheSizeBytes / sizeof(float) / stride;
  
  float *d_out;
  CUDA_CALL(cudaMallocAsync(&d_out, out_size * sizeof(float), args->stream));
  
  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  l1_store_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (d_out, out_size, stride);
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);

  double per_thread_bandwidth = L1_STORE_STEPS * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s {:d} ms", FUNC_NAME(l1_store_kernel), args->id, bandwidth, (int)elapsed_ms);
  
  // cleanup
  CUDA_CALL(cudaFree(d_out));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}