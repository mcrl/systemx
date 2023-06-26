#pragma once

#define KERNELS() \
  T(idle)

// struct GPU_CONFIG {
//   int nSMS;
//   int nThreadsPerWarp;
//   int maxThreadsPerSM;
//   int maxWarpsPerSM;
//   int maxThreadBlocksPerSM;
//   int maxThreadBlockSize;
//   int sharedMemoryPerSM;
//   int maxRegistersPerSM;
//   int maxRegistersPerThreadBlock;
//   int maxRegistersPerThread;
// };

// GPU_CONFIG V100_CONFIG = {
//   nSMs = 80,
//   nThreadsPerWarp = 32,
//   maxThreadsPerSM = 2048,
//   maxWarpsPerSM = 64,
//   maxThreadBlocksPerSM = 32,
//   maxThreadBlockSize = 1024
//   sharedMemoryPerSM = 96 * 1024, // 96 KB
//   maxRegistersPerSM = 65536,
//   maxRegistersPerThreadBlock = 65536,
//   maxRegistersPerThread = 255
// };