#pragma once

#include <pthread.h>

#include <vector>
#include <string>
#include <functional>
#include <map>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "kernels.hpp"

namespace SYSTEMX {
namespace core {

class Driver {
public:
  Driver(int gpu_index);
  ~Driver();
  cudaStream_t getStream(uint stream_id);
  cudaStream_t createStream(); // TODO: deprecated
  void launchKernel(std::string kernel, kernel_run_args args);
  void *mallocDBuf(size_t size, cudaStream_t stream);
  void setDBuf(void *ptr, int value, size_t count, cudaStream_t stream);
  cublasHandle_t createCublasHandle();
  cudaDeviceProp device_properties_;
#define T(op) void op##Run();
  KERNELS()
#undef T

private:
  int gpu_index_;
  // std::vector<cudaStream_t> streams_;
  std::map<uint, cudaStream_t> stream_map_;
  std::map<std::string, std::function<void(void)>> kernel_map_;
  std::vector<pthread_t> threads_;
  std::vector<void *> dbufs_;
  std::vector<cublasHandle_t> cublas_handles_;
  void freeDBuf(void *ptr);
};
}
}