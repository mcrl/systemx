#include "driver.hpp"

#include <functional>
#include <map>
#include <string>
#include <thread>

#include "spdlog/spdlog.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "utils.hpp"

namespace SYSTEMX {
namespace core {

Driver::Driver(uint gpu_index) {
  spdlog::debug("Creating driver {}", gpu_index);
  gpu_index_ = gpu_index;
  CUDA_CALL(cudaSetDevice(gpu_index_));
  CUDA_CALL(cudaGetDeviceProperties(&device_properties_, gpu_index_));

#define T(op) kernel_map_[#op] = [&](kernel_run_args *args){return this->op##Run(args);};
  KERNELS()
#undef T
}

Driver::~Driver() {
  spdlog::debug("Destroying driver {}", gpu_index_);
  CUDA_CALL(cudaSetDevice(gpu_index_));

  for (std::thread &t : threads_) {
    t.join();
  }
  
  for (const auto &[id, stream] : stream_map_) {
    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaStreamDestroy(stream));
  }
}

cudaStream_t Driver::getStream(uint stream_id, int stream_priority = 0) {
  if (stream_map_.find(stream_id) == stream_map_.end()) {
    int greatest, least; // range of meaningful priorities: [*greatest, *least]
                        // smaller number represents higher priority
    CUDA_CALL(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    if (stream_priority < greatest || stream_priority > least) {
      throw runtime_error("Stream priority out of range");
    }
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, stream_priority));
    stream_map_[stream_id] = stream;
  } else {
    int fixed_priority;
    CUDA_CALL(cudaStreamGetPriority(stream_map_[stream_id], &fixed_priority));
    if (fixed_priority != stream_priority) {
      throw runtime_error("Attempt to modify existing stream's priority");
    }
  }
  return stream_map_[stream_id];
}

void Driver::assertDeviceCorrect() {
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  if ((uint)device != gpu_index_) {
    throw runtime_error("Device not correct");
  }
}

void Driver::launchKernel(std::string kernel, kernel_run_args *kargs) {
  if (kernel_map_.find(kernel) == kernel_map_.end()) {
    throw runtime_error("Kernel not found");
  }
  spdlog::debug("Launching kernel {} on driver {}", kernel, gpu_index_);
  // bind Driver::gpu_index_ to thread & launch thread kernel 
  std::thread t([&,kernel](kernel_run_args *args) {
    CUDA_CALL(cudaSetDevice(this->gpu_index_));
    this->kernel_map_[kernel](args);
  }, kargs);
  threads_.push_back(std::move(t));
}
}
}