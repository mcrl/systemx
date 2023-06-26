#pragma once

#include <vector>
#include <string>
#include <functional>
#include <map>

#include "cuda_runtime.h"

#include "kernels.hpp"

namespace SYSTEMX {
namespace core {

class Driver {
public:
  Driver(int gpu_index);
  ~Driver();
  void launchKernel(std::string kernel);

#define T(op) void op##Run();
  KERNELS()
#undef T

private:
  int gpu_index_;
  std::vector<cudaStream_t> streams_;
  std::map<std::string, std::function<void(void)>> kernel_map_;
  cudaStream_t createStream();
};
}
}