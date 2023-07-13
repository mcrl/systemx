// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"
#include "utils.hpp"

#define PCIE_STORE_NUM_FLOATS_PER_STEP 800000000 // 3.2GB
#define PCIE_READ_NUM_FLOATS_PER_STEP 800000000 // 3.2GB
#define P2P true // TODO: make this as a kargs option 
#define TRANSFER_TYPE float4

typedef enum {
  CE = 0,
  SM = 1,
} P2PEngine;

P2PEngine p2p_mechanism = SM; // By default use SM initiated p2p transfers
                              // TODO: add support for CE initiated p2p transfers
using SYSTEMX::core::Driver;

// This kernel is for demonstration purposes only, not a performant kernel for p2p transfers.
// num_elems is the number of T in dest (and src)
template<typename T>
__global__ void copyp2p_kernel(T *__restrict__ dest, T const *__restrict__ src,
                               size_t num_elems, const uint steps) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

#pragma unroll
  for (uint i = 0; i < steps; i++) {
#pragma unroll(5)
    for (size_t j = globalId; j < num_elems; j += stride) {
      dest[j] = src[j];
    }
  }
}

void Driver::pcieReadRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  assertDeviceCorrect();

  // create event to check if buffer is ready
  cudaEvent_t ready;
  CUDA_CALL(cudaEventCreate(&ready));
  
  // set appropriate device buffer
  const int in_size = PCIE_READ_NUM_FLOATS_PER_STEP;  
  CUDA_CALL(cudaMallocAsync(&((*((args->shared_buffer_map)->at("d_in")))[gpu_index_]),
                            in_size * sizeof(float),
                            args->stream));
  CUDA_CALL(cudaEventRecord(ready, args->stream));

  // check if all buffers are ready
  CUDA_CALL(cudaEventSynchronize(ready));

  // check if all gpus are ready
  (*(args->shared_counter_map))["deviceBufferReady"]->decrement();

  // start kernel
  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));

  // work per thread: float4 * (in_size / 4 / (dimGrid.x * dimBlock.x))
  const int intra_step_access_per_thread = in_size / (args->dimGrid.x * args->dimBlock.x);

  // TODO: make multiple srcs possible
  int access = 0;
  CUDA_CALL(cudaDeviceCanAccessPeer(&access, gpu_index_, gpu_index_ == 0 ? 1 : 0));
  
  if (P2P && access && p2p_mechanism == SM) {
    copyp2p_kernel<TRANSFER_TYPE> << <args->dimGrid, args->dimBlock, 0, args->stream >> > (
      (TRANSFER_TYPE *)(*((args->shared_buffer_map)->at("d_in")))[gpu_index_],
      (TRANSFER_TYPE *)(*((args->shared_buffer_map)->at("d_in")))[gpu_index_ == 0 ? 1 : 0],
      in_size / (sizeof(TRANSFER_TYPE) / sizeof(float)), args->steps);
  } else {
    cudaMemcpyPeerAsync(
      (*((args->shared_buffer_map)->at("d_in")))[gpu_index_], gpu_index_,
      (*((args->shared_buffer_map)->at("d_in")))[gpu_index_ == 0 ? 1 : 0], gpu_index_ == 0 ? 1 : 0,
      in_size * sizeof(float), args->stream);
  }
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);
  double per_thread_bandwidth = args->steps * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s {:d} ms", FUNC_NAME(copyp2p_kernel), args->id, bandwidth, (int)elapsed_ms);

  // check if all gpus are finished
  (*(args->shared_counter_map))["deviceKernelFinish"]->decrement();
  
  // cleanup
  CUDA_CALL(cudaFree((*((args->shared_buffer_map)->at("d_in")))[gpu_index_]));
  CUDA_CALL(cudaEventDestroy(ready));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}

void Driver::pcieWriteRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  assertDeviceCorrect();

  // create event to check if buffer is ready
  cudaEvent_t ready;
  CUDA_CALL(cudaEventCreate(&ready));
  
  // set appropriate device buffer
  const int in_size = PCIE_STORE_NUM_FLOATS_PER_STEP;  
  CUDA_CALL(cudaMallocAsync(&((*((args->shared_buffer_map)->at("d_in")))[gpu_index_]),
                            in_size * sizeof(float),
                            args->stream));
  CUDA_CALL(cudaEventRecord(ready, args->stream));

  // check if all buffers are ready
  CUDA_CALL(cudaEventSynchronize(ready));

  // check if all gpus are ready
  (*(args->shared_counter_map))["deviceBufferReady"]->decrement();

  // start kernel
  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));

  // work per thread: float4 * (in_size / 4 / (dimGrid.x * dimBlock.x))
  const int intra_step_access_per_thread = in_size / (args->dimGrid.x * args->dimBlock.x);

  // TODO: make multiple dests possible
  int access = 0;
  CUDA_CALL(cudaDeviceCanAccessPeer(&access, gpu_index_, gpu_index_ == 0 ? 1 : 0));
  if (P2P && access && p2p_mechanism == SM) {
    copyp2p_kernel<TRANSFER_TYPE> << <args->dimGrid, args->dimBlock, 0, args->stream >> > (
      (TRANSFER_TYPE *)(*((args->shared_buffer_map)->at("d_in")))[gpu_index_ == 0 ? 1 : 0],
      (TRANSFER_TYPE *)(*((args->shared_buffer_map)->at("d_in")))[gpu_index_],
      in_size / (sizeof(TRANSFER_TYPE) / sizeof(float)), args->steps);
  } else {
    cudaMemcpyPeerAsync(
      (*((args->shared_buffer_map)->at("d_in")))[gpu_index_ == 0 ? 1 : 0], gpu_index_ == 0 ? 1 : 0,
      (*((args->shared_buffer_map)->at("d_in")))[gpu_index_], gpu_index_,
      in_size * sizeof(float), args->stream);
  }
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);
  double per_thread_bandwidth = args->steps * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s {:d} ms", FUNC_NAME(copyp2p_kernel), args->id, bandwidth, (int)elapsed_ms);

  // check if all gpus are finished
  (*(args->shared_counter_map))["deviceKernelFinish"]->decrement();
  
  // cleanup
  CUDA_CALL(cudaFree((*((args->shared_buffer_map)->at("d_in")))[gpu_index_]));
  CUDA_CALL(cudaEventDestroy(ready));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}