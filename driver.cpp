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

Driver::Driver(int gpu_index) {
  gpu_index_ = gpu_index;
  CUDA_CALL(cudaSetDevice(gpu_index_));
  CUDA_CALL(cudaGetDeviceProperties(&device_properties_, gpu_index_));

#define T(op) kernel_map_[#op] = [&](kernel_run_args *args){return this->op##Run(args);};
  KERNELS()
#undef T
}

Driver::~Driver() {
  spdlog::debug("Destroying driver");

  for (std::thread &t : threads_) {
    t.join();
  }
  
  spdlog::debug("Destroying streams");
  for (const auto &[id, stream] : stream_map_) {
    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaStreamDestroy(stream));
  }
  spdlog::debug("Destroying device buffers");
  for (void *ptr : dbufs_) {
    CUDA_CALL(cudaFree(ptr));
  }
  spdlog::debug("Destroying cuBLAS handles");
  for (cublasHandle_t handle : cublas_handles_) {
    CUBLAS_CALL(cublasDestroy(handle));
  }
}

cudaStream_t Driver::getStream(uint stream_id) {
  if (stream_map_.find(stream_id) == stream_map_.end()) {
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    stream_map_[stream_id] = stream;
  }
  return stream_map_[stream_id];
}

void *Driver::mallocDBuf(size_t size, cudaStream_t stream) {
  void *ptr;
  CUDA_CALL(cudaMallocAsync(&ptr, size, stream));
  dbufs_.push_back(ptr);
  return ptr;
}

// Explicitly called at destructor to free all device buffers
void Driver::freeDBuf(void *ptr) {
  CUDA_CALL(cudaFree(ptr));
}

void Driver::setDBuf(void *ptr, int value, size_t count, cudaStream_t stream) {
  CUDA_CALL(cudaMemsetAsync(ptr, value, count, stream));
}

cublasHandle_t Driver::createCublasHandle() {
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  cublas_handles_.push_back(handle);
  return handle;
}

void Driver::launchKernel(std::string kernel, kernel_run_args *kargs) {
  if (kernel_map_.find(kernel) == kernel_map_.end()) {
    throw runtime_error("Kernel not found");
  }

  std::thread t(kernel_map_[kernel], kargs);
  threads_.push_back(std::move(t));
}
}
}