#pragma once

#include "cuda_runtime.h"

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