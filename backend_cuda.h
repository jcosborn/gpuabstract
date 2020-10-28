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

template <typename F, typename Arg, typename... Dims>
__global__ void Kernel1d(F &f, Arg &arg, Dims... ndi)
{
  unsigned int i = globalIdx_x;
  unsigned int gridSize = globalDim_x;
  while(i < arg.threads.x) {
    f(i);
    i += gridSize;
  }
}


struct Kern1d {
  template <template <typename> class Functor, typename Arg>
  void launch_host(const qudaStream_t &stream, const Arg &arg) const
  {
    Functor<Arg> f(const_cast<Arg &>(arg));
    for (int i = 0; i < (int)arg.threads.x; i++) {
      f(i);
    }
  }
  template <template <typename> class Functor, typename Arg>
  void launch_device(const qudaStream_t &stream, const Arg &arg) const
  {
    Functor<Arg> f(const_cast<Arg &>(arg));
    auto nthreads = 32;
    auto s = const_cast<qudaStream_t &>(stream);
    qudaLaunch(2, nthreads, 0, s, Kernel1d, f, arg);
  }
  template <template <typename> class Functor, typename Arg>
  void launch(const qudaStream_t &stream, const Arg &arg) const
  {
    //launch_host<Functor, Arg>(stream, arg);
    launch_device<Functor, Arg>(stream, arg);
  }
};
