#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"

using SYSTEMX::core::Driver;

__global__ void idle_kernel(uint milliseconds) {
  for (uint i = 0; i < milliseconds; ++i) {
    __nanosleep(1000000U);
  }
}

void Driver::idleRun() {
  spdlog::info(__PRETTY_FUNCTION__);

  cudaStream_t stream = createStream();

  uint milliseconds = 100;

  // TODO: use static values from gpu configs
  const int maxThreadBlockSize = 1024;
  const int maxThreadsPerSM = 2048;
  const int nSMs = 80;
  
  dim3 gridDim(maxThreadsPerSM/maxThreadBlockSize, nSMs/2, 1); // Fully occupy half of total SMs
  dim3 blockDim(maxThreadBlockSize, 1, 1);
  idle_kernel<<<gridDim, blockDim, 0, stream>>>(milliseconds);
}