#include "spdlog/spdlog.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "driver.hpp"
#include "kernels.hpp"

using SYSTEMX::core::Driver;

// TODO: 
//  - Set appropriate dimension sizes and SM count according to args->dimGrid/dimBlock
void Driver::cublasGemmRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = args->stream;
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSetStream(handle, stream));

  float *d_A, *d_B, *d_C;
  uint64_t M = 8192, K = 8192, N = 8192;

  CUDA_CALL(cudaMallocAsync(&d_A, M * K * sizeof(float), stream));
  CUDA_CALL(cudaMallocAsync(&d_B, K * N * sizeof(float), stream));
  CUDA_CALL(cudaMallocAsync(&d_C, M * N * sizeof(float), stream));

  CUDA_CALL(cudaMemsetAsync(d_A, 1, M * K * sizeof(float), stream));
  CUDA_CALL(cudaMemsetAsync(d_B, 1, K * N * sizeof(float), stream));
  CUDA_CALL(cudaMemsetAsync(d_C, 0, M * N * sizeof(float), stream));
    
  // Launch non-blocking compute
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  // CUBLAS_CALL(cublasSetSmCountTarget(handle, 60)); // set cublas SM count

  cudaEvent_t start, end;
  start = std::get<1>(args->events[0]);
  end = std::get<1>(args->events[1]);

  float elapsed_ms;
  CUDA_CALL(cudaEventRecord(start, args->stream));
  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
  CUDA_CALL(cudaEventRecord(end, args->stream));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, end));

  double gflops = 2.0 * M * K * N / elapsed_ms * 1e3 / 1e9;
  spdlog::info("{}(id: {}) {:.2f} Gflops {:d} ms", "cublasGemm", args->id, gflops, elapsed_ms);

  // cleanup
  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C));
  CUBLAS_CALL(cublasDestroy(handle));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(end));
}