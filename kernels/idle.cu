#include <tuple>

#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"
#include "utils.hpp"

using SYSTEMX::core::Driver;

__global__ void idle_kernel(uint milliseconds) {
  for (uint i = 0; i < milliseconds; ++i) {
    __nanosleep(1000000U);
  }
}

void Driver::idleRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  assertDeviceCorrect();

  uint milliseconds = 300;

  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  CUDA_CALL(cudaEventRecord(start, args->stream));
  idle_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (milliseconds);
  CUDA_CALL(cudaEventRecord(end, args->stream));

  float elapsed_ms;
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));
  spdlog::info("{}(id:{:d}) {:d} ms", FUNC_NAME(idle_kernel), args->id, elapsed_ms);

  // cleanup
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}