#pragma once

#define KERNELS() \
  T(idle) \
  T(registerCompute) \
  T(gmemLoad) \
  T(gmemStore) \
  T(l2Load) \
  T(l2Store) \

inline __device__ float mad(const float a, const float b, const float c) {
  return a * b + c;
}

inline __device__ float int2floatCast(const int i) {
  return static_cast<float>(i);
}
