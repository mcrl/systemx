// #include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"

using SYSTEMX::core::Driver;

__global__ void idle_kernel(uint seconds) {
  for (uint i = 0; i < seconds; ++i) {
    for (int j = 0; j < 1000; j++) {
      __nanosleep(1000000U);
    }
  }
}

void Driver::idle_run() {
  // spdlog::trace("Driver::idle_run()");

  cudaStream_t stream = createStream();

  uint idle_seconds = 10;
  
  dim3 gridDim(64, 64, 1);
  dim3 blockDim(1, 1, 1);
  idle_kernel<<<gridDim, blockDim, 0, stream>>>(idle_seconds);
}