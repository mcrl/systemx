#include "driver.hpp"

#include <functional>
#include <map>

#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "utils.hpp"

namespace SYSTEMX {
namespace core {

Driver::Driver(int gpu_index) {
  gpu_index_ = gpu_index;
  CUDA_CALL(cudaSetDevice(gpu_index_));

// #define T(op) kernel_map_[#op] = std::bind(&Driver::op##Run, this);
#define T(op) kernel_map_[#op] = [&]{this->op##Run();};
  KERNELS()
#undef T
}

Driver::~Driver() {
  spdlog::info("Destroying driver");
  
  for (cudaStream_t stream : streams_) {
    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaStreamDestroy(stream));
  }
}

cudaStream_t Driver::createStream() {
  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreate(&stream));
  streams_.push_back(stream);
  return stream;
}

void Driver::launchKernel(string kernel) {
  spdlog::info("Launching kernel: {0}", kernel);

  if (kernel_map_.find(kernel) == kernel_map_.end()) {
    throw runtime_error("Kernel not found");
  }
  kernel_map_[kernel]();
}
}
}