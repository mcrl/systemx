// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"
#include "utils.hpp"

#define PCIE_STORE_NUM_ELEMS_PER_STEP 1024 * 1024
#define P2P true // TODO: make this as a kargs option 

typedef enum {
  CE = 0,
  SM = 1,
} P2PEngine;

P2PEngine p2p_mechanism = SM; // By default use SM initiated p2p transfers
                              // TODO: add support for CE initiated p2p transfers
using SYSTEMX::core::Driver;

// This kernel is for demonstration purposes only, not a performant kernel for
// p2p transfers.
__global__ void copyp2p_kernel(float *__restrict__ dest, const float *__restrict__ src,
                               size_t num_elems, const int stride, const uint steps) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll(5)
  for (uint i = 0; i < steps; i++) {
    for (size_t j = globalId; j < num_elems; j += stride) {
      dest[j] = src[j];
    }
  }
}

// TODO: implement
void Driver::pcieReadRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  assertDeviceCorrect();

  const int intra_step_access_per_thread = 0; // TODO: fill in

  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  // pcie_read_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (d, d_size, stride, args->steps);
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);

  double per_thread_bandwidth = args->steps * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s {:d} ms", FUNC_NAME(pcie_read_kernel), args->id, bandwidth, (int)elapsed_ms);
  
  // cleanup
  // CUDA_CALL(cudaFree(d));
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
  const int in_size = PCIE_STORE_NUM_ELEMS_PER_STEP;  
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

  const int stride = args->dimGrid.x * args->dimBlock.x;
  const int intra_step_access_per_thread = in_size / stride;
  
  // TODO: make multiple dests possible
  int access = 0;
  CUDA_CALL(cudaDeviceCanAccessPeer(&access, gpu_index_, gpu_index_ == 0 ? 1 : 0));
  if (P2P && access && p2p_mechanism == SM) {
    copyp2p_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > ((*((args->shared_buffer_map)->at("d_in")))[gpu_index_ == 0 ? 1 : 0],
                                                                            (*((args->shared_buffer_map)->at("d_in")))[gpu_index_],
                                                                            in_size, stride, args->steps);
  } else {
    cudaMemcpyPeerAsync((*((args->shared_buffer_map)->at("d_in")))[gpu_index_ == 0 ? 1 : 0], gpu_index_ == 0 ? 1 : 0,
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