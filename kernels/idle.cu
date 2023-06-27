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

void Driver::idleRun() {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = createStream();

  uint milliseconds = 100;

  const int maxThreadsPerBlock = device_properties_.maxThreadsPerBlock;
  const int maxThreadsPerMultiProcessor = device_properties_.maxThreadsPerMultiProcessor;
  const int multiProcessorCount = device_properties_.multiProcessorCount;

  // Fully occupy half of total SMs
  dim3 gridDim((maxThreadsPerMultiProcessor / maxThreadsPerBlock) * (multiProcessorCount / 2), 1, 1);
  dim3 blockDim(maxThreadsPerBlock, 1, 1);
  idle_kernel << <gridDim, blockDim, 0, stream >> > (milliseconds);
}