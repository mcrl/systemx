// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"
#include "utils.hpp"

#define PCIE_STORE_NUM_ELEMS_PER_STEP 1024 * 1024

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
                               size_t num_elems, const uint steps) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;
  
#pragma unroll(5)
  for (size_t i = globalId; i < num_elems; i += gridSize) {
    dest[i] = src[i];
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

  // check p2p access
  // TODO: For now, we assume that all gpus can either access all others via p2p or not at all
  bool p2p = true;
  for (uint peer_gpu : args->gpus) {
    if (peer_gpu == gpu_index_) continue;
    int access;
    CUDA_CALL(cudaDeviceCanAccessPeer(&access, gpu_index_, peer_gpu)); // 1 if gpu_index_ -> peer_gpu is possible
    if (access) {
      spdlog::debug("GPU {} can directly access GPU {} via p2p", gpu_index_, peer_gpu);
      CUDA_CALL(cudaDeviceEnablePeerAccess(peer_gpu, 0));
    } else {
      spdlog::warn("GPU {} cannot directly access GPU {} via p2p. Fallback to cudaMemcpyPeerAsync", gpu_index_, peer_gpu);
      p2p = false;
    }
  }

  // check if all buffers are ready
  CUDA_CALL(cudaEventSynchronize(ready));

  // check if all gpus are ready
  (*(args->shared_counter_map))["deviceBufferReady"]->decrement();

  // start kernel
  const int intra_step_access_per_thread = 0; // TODO: fill in

  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));

  // TODO: make multiple dests possible
  printf("[GPU%d] %p -> %p (after ready)\n", gpu_index_, (*((args->shared_buffer_map)->at("d_in")))[gpu_index_], (*((args->shared_buffer_map)->at("d_in")))[gpu_index_ == 0 ? 1 : 0]);

  if (p2p && p2p_mechanism == SM) {
    copyp2p_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > ((*((args->shared_buffer_map)->at("d_in")))[gpu_index_ == 0 ? 1 : 0],
                                                                            (*((args->shared_buffer_map)->at("d_in")))[gpu_index_],
                                                                            in_size, args->steps);
  } else {
    // TODO: make multiple dests possible
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