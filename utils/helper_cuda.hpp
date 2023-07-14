#pragma once

#include <stdio.h>

#include <cstdlib>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#ifndef CUDA_CALL
#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n",                         \
               __FILE__, __LINE__,                                             \
              err, cudaGetErrorString(err));                                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }                                                                            
#endif

#ifndef CUBLAS_CALL
#define CUBLAS_CALL(f)                                      \
  {                                                         \
    cublasStatus_t err = (f);                               \
    if (err != CUBLAS_STATUS_SUCCESS) {                     \
      fprintf(stderr, "cuBLAS error at [%s:%d] %d %s\n",    \
               __FILE__, __LINE__,                          \
              err, cublasGetStatusString(err));             \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  }
#endif

namespace SYSTEMX {
namespace utils {
void enable_device_memory_access(int device, int peer);
void disable_device_memory_access(int device, int peer);
void check_pointer_attributes(int device, void *ptr);
}
}