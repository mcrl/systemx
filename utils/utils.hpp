#pragma once

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "timer.hpp"
#include "helper_string.hpp"

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
      exit(EXIT_FAILURE);                                              \
    }                                                       \
  }
#endif