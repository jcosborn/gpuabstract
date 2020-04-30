#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define qudaError_t cudaError_t
#define qudaDeviceProp cudaDeviceProp
#define qudaStream_t cudaStream_t
#define qudaStreamCreate cudaStreamCreate
#define qudaStreamSynchronize cudaStreamSynchronize
#define qudaGetDeviceProperties cudaGetDeviceProperties

#define threadIdx_x threadIdx.x
#define blockDim_x blockDim.x
#define blockIdx_x blockIdx.x
#define gridDim_x gridDim.x
#define globalIdx_x (blockIdx.x * (blockDim.x) + threadIdx.x)
#define globalDim_x (gridDim.x * blockDim.x)

#define qudaLaunch(gridDim0, blockDim0, sharedMem0, stream0, func0, ...) \
  func0<<<gridDim0,blockDim0,sharedMem0,stream0>>>(__VA_ARGS__);

void *
qudaMallocManaged(size_t size)
{
  void *p;
  auto err = cudaMallocManaged(&p, size);
  if(err!=0) {
    std::cerr << "cudaMallocManaged error: " << err << std::endl;
  }
  return p;
}
