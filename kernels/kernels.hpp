#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <condition_variable>
#include <mutex>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "utils.hpp"

#define KERNELS() \
  T(idle)         \
  T(aluCompute)   \
  T(gmemLoad)     \
  T(gmemStore)    \
  T(l2Load)       \
  T(l2Store)      \
  T(l1Load)       \
  T(l1Store)      \
  T(pcieRead)     \
  T(pcieWrite)    \
  T(cublasGemm)   \

#define FUNC_NAME(f) #f

typedef std::tuple<std::string, cudaEvent_t> event_tuple_t;

struct kernel_run_args {
  uint id;
  cudaStream_t stream;
  std::vector<uint> gpus;
  dim3 dimGrid;
  dim3 dimBlock;
  uint steps;
  std::vector<event_tuple_t> events;

  /* optional private arguments */
  std::vector<uint> interactions; // list of peer driver (:=gpu) indices that the driver
                                  // of which this kernel is launched interacts with

  /* optional shared pointers */
  SYSTEMX::utils::shared_counter_map_t *shared_counter_map;
  SYSTEMX::utils::shared_buffer_map_t *shared_buffer_map;
};

inline __device__ float mad(const float a, const float b, const float c) {
  return a * b + c;
}

inline __device__ float int2floatCast(const int i) {
  return static_cast<float>(i);
}

// Query the thread's allocated SM ID
inline __device__ void sm_id(uint32_t *smid) {
  asm volatile("mov.u32 %0, %%smid;" : "=r"(*smid));
}

inline uint get_nthreads(dim3 dimGrid, dim3 dimBlock) {
  return dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x * dimBlock.y * dimBlock.z;
}

inline uint get_nblocks(dim3 dimGrid) {
  return dimGrid.x * dimGrid.y * dimGrid.z;
}