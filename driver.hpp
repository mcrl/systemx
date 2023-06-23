#pragma once

#include <vector>

#include "cuda_runtime.h"

#include "kernels.hpp"

namespace SYSTEMX {
namespace core {

class Driver {
public:
  Driver(int _gpu_index);
  ~Driver();
  cudaStream_t createStream();

#define T(op) void op##_run();
  KERNELS()
#undef T

private:
  int gpu_index;
  std::vector<cudaStream_t> streams;
};
}
}