#include "spdlog/spdlog.h"
#include "cuda_runtime.h"

#include "driver.hpp"
#include "kernels.hpp"

using SYSTEMX::core::Driver;

__global__ void pcie_read_kernel(float *d, const int d_size,
                               const int stride, const uint steps) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  
}

__global__ void pcie_write_kernel(float *d, const int d_size,
                                 const int stride, const uint steps) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

}

void Driver::pcieReadRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  assertDeviceCorrect();

  const int intra_step_access_per_thread = 0; // TODO: fill in

  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  // pcie_read_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (d, d_size, stride, args->steps);
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);

  double per_thread_bandwidth = args->steps * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s {:d} ms", FUNC_NAME(pcie_read_kernel), args->id, bandwidth, (int)elapsed_ms);
  
  // cleanup
  // CUDA_CALL(cudaFree(d));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}

void Driver::pcieWriteRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  assertDeviceCorrect();

  const int intra_step_access_per_thread = 0; // TODO: fill in

  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  // pcie_write_kernel << <args->dimGrid, args->dimBlock, 0, args->stream >> > (d, d_size, stride, args->steps);
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  const int total_threads = get_nthreads(args->dimGrid, args->dimBlock);

  double per_thread_bandwidth = args->steps * intra_step_access_per_thread * sizeof(float) / elapsed_ms / 1e6;
  double bandwidth = per_thread_bandwidth * total_threads;
  spdlog::info("{}(id: {}) {:.2f} GB/s {:d} ms", FUNC_NAME(pcie_write_kernel), args->id, bandwidth, (int)elapsed_ms);
  
  // cleanup
  // CUDA_CALL(cudaFree(d));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}