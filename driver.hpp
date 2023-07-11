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
  cudaStream_t getStream(uint stream_id, int stream_priority);
  void launchKernel(std::string kernel, kernel_run_args *kargs);
  void assertDeviceCorrect();
  cudaDeviceProp device_properties_;
  static uint ngpus_; // number of gpus in the node
#define T(op) void op##Run(kernel_run_args *args);
  KERNELS()
#undef T

private:
  uint gpu_index_; // gpu index of this driver
  std::map<std::string, std::function<void(kernel_run_args *)>> kernel_map_;
  std::vector<std::thread> threads_;
  std::map<uint, cudaStream_t> stream_map_; // key is set to the "logical" stream id 
                                            // given from benchmark json, not the
                                            // "physical" cuda stream id which can be
                                            // queried with `cudaStreamGetId`
};
}
}