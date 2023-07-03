#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"

using SYSTEMX::core::Driver;

__global__ void idle_kernel(uint milliseconds) {
  for (uint i = 0; i < milliseconds; ++i) {
    __nanosleep(1000000U);
  }
}

void Driver::idleRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  uint milliseconds = 300;
  // TODO: add events
  idle_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (milliseconds);
}