#pragma once

#include <vector>
#include <string>
#include <functional>
#include <map>
#include <thread>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "kernels.hpp"

namespace SYSTEMX {
namespace core {

class Driver {
public:
  Driver(uint gpu_index);
  ~Driver();
  cudaStream_t getStream(uint stream_id);
  void launchKernel(std::string kernel, kernel_run_args *kargs);
  void assertDeviceCorrect();
  cudaDeviceProp device_properties_;
#define T(op) void op##Run(kernel_run_args *args);
  KERNELS()
#undef T

private:
  uint gpu_index_;
  std::map<std::string, std::function<void (kernel_run_args *)>> kernel_map_;
  std::map<uint, cudaStream_t> stream_map_;
  std::vector<std::thread> threads_;
};
}
}