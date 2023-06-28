#pragma once

#define KERNELS() \
  T(idle) \
  T(registerCompute) \
  T(gmemLoad) \
  T(gmemStore) \
  T(l2Load) \
  T(cublasGemm) \

inline __device__ float mad(const float a, const float b, const float c) {
  return a * b + c;
}

inline __device__ float int2floatCast(const int i) {
  return static_cast<float>(i);
}

// Query the thread's allocated SM ID
inline __device__ void sm_id(uint32_t *smid) {
  asm volatile("mov.u32 %0, %%smid;" : "=r"(*smid));
}