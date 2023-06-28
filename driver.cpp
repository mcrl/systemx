#include "driver.hpp"

#include <functional>
#include <map>

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

// #define T(op) kernel_map_[#op] = std::bind(&Driver::op##Run, this);
#define T(op) kernel_map_[#op] = [&]{this->op##Run();};
  KERNELS()
#undef T
}

Driver::~Driver() {
  spdlog::debug("Destroying driver");
  
  spdlog::debug("Destroying streams");
  for (cudaStream_t stream : streams_) {
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

cudaStream_t Driver::createStream() {
  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreate(&stream));
  streams_.push_back(stream);
  return stream;
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

void Driver::launchKernel(string kernel) {
  if (kernel_map_.find(kernel) == kernel_map_.end()) {
    throw runtime_error("Kernel not found");
  }
  kernel_map_[kernel]();
}
}
}