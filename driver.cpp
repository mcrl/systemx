#include "driver.hpp"

#include "cuda_runtime.h"

#include "utils.hpp"

namespace SYSTEMX {
namespace core {

Driver::Driver(int _gpu_index) {
  gpu_index = _gpu_index;
  CUDA_CALL(cudaSetDevice(gpu_index));
}

Driver::~Driver() {
  for (cudaStream_t stream : streams) {
    CUDA_CALL(cudaStreamDestroy(stream));
  }
}

cudaStream_t Driver::createStream() {
  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreate(&stream));
  streams.push_back(stream);
  return stream;
}
}
}