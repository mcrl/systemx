#include "spdlog/spdlog.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "driver.hpp"
#include "kernels.hpp"
#include "utils.hpp"

using SYSTEMX::core::Driver;

// TODO: Refactor
void Driver::cublasGemmRun(kernel_run_args *args) {
  spdlog::trace(__PRETTY_FUNCTION__);

  cudaStream_t stream = args->stream;

  cublasHandle_t handle = createCublasHandle();
  CUBLAS_CALL(cublasSetStream(handle, stream));

  float *d_A, *d_B, *d_C;
  int M = 8192, K = 8192, N = 8192;

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
  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
}