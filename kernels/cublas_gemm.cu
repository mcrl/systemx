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
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSetStream(handle, stream));

  float *d_A, *d_B, *d_C;
  int M = args->dimGrid.x * args->dimGrid.y * args->dimGrid.z * args->dimBlock.x * args->dimBlock.y * args->dimBlock.z,
    K = args->dimGrid.x * args->dimGrid.y * args->dimGrid.z * args->dimBlock.x * args->dimBlock.y * args->dimBlock.z,
    N = args->dimGrid.x * args->dimGrid.y * args->dimGrid.z * args->dimBlock.x * args->dimBlock.y * args->dimBlock.z;

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

  CUDA_CALL(cudaEventRecord(start, args->stream));
  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
  CUDA_CALL(cudaEventRecord(end, args->stream));

  float elapsed_time = 0;
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, end));
  spdlog::info("Kernel {} took {} ms", FUNC_NAME(idle_kernel), elapsed_time);
  
  // cleanup
  CUBLAS_CALL(cublasDestroy(handle));
}