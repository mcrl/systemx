#include "helper_cuda.hpp"

#include "cuda_runtime.h"

namespace SYSTEMX {
namespace utils {

// wrapper to enable peer device memory access
void enable_device_memory_access(int device, int peer) {
  CUDA_CALL(cudaSetDevice(device));
  
  // enable p2p access
  int access = 0;
  CUDA_CALL(cudaDeviceCanAccessPeer(&access, device, peer));
  if (access) {
    CUDA_CALL(cudaDeviceEnablePeerAccess(peer, 0));
  }
  
  // enable access to entire pool of peer memory
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, peer);
  cudaMemAccessDesc desc = {};
  desc.location.type = cudaMemLocationTypeDevice;
  desc.location.id = device;
  desc.flags = cudaMemAccessFlagsProtReadWrite;
  cudaMemPoolSetAccess(mempool, &desc, 1 /* numDescs */);
}

// wrapper to disable peer device memory access
void disable_device_memory_access(int device, int peer) {
  CUDA_CALL(cudaSetDevice(device));

  // disable p2p access
  int access = 0;
  CUDA_CALL(cudaDeviceCanAccessPeer(&access, device, peer));
  if (access) {
    CUDA_CALL(cudaDeviceDisablePeerAccess(peer));
  }
  
  // disable access to entire pool of peer memory
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, peer);
  cudaMemAccessDesc desc = {};
  desc.location.type = cudaMemLocationTypeDevice;
  desc.location.id = device;
  desc.flags = cudaMemAccessFlagsProtNone;
  cudaMemPoolSetAccess(mempool, &desc, 1 /* numDescs */);
}

// wrapper to check attributes of a pointer w.r.t. current device
void check_pointer_attributes(int device, void *ptr) {
  CUDA_CALL(cudaSetDevice(device));

  cudaPointerAttributes attr;
  CUDA_CALL(cudaPointerGetAttributes(&attr, ptr));
  printf("Pointer %p is dev: %d, devp: %p, hostp: %p\n", ptr, attr.device, attr.devicePointer, attr.hostPointer);
}
}
}