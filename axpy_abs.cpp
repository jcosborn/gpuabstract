#include <iostream>
#include "backends.h"

template <typename TX, typename TY, typename TF>
struct Arg {
  TX x;
  TY y;
  TF f;
  const int length;
  Arg(TX x, TY y, TF f, int length): x(x), y(y), f(f), length(length) {
    ;
  }
};
template <typename TR, typename TX, typename TF>
auto makeArg(TR r, TX x, TF f, int length) {
  return Arg<TR,TX,TF>(r, x, f, length);
}

template <typename Arg, typename... Dims>
__global__ void Kernel(Arg arg, Dims... ndi)
{
  //unsigned int i = blockIdx.x * (blockDim.x) + threadIdx.x;
  //unsigned int gridSize = gridDim.x * blockDim.x;
  //unsigned int i = blockIdx_x * (blockDim_x) + threadIdx_x;
  //unsigned int gridSize = gridDim_x * blockDim_x;
  unsigned int i = globalIdx_x;
  unsigned int gridSize = globalDim_x;
  while(i < arg.length) {
    arg.f(arg.x[i], arg.y[i]);
    i += gridSize;
  }
}

template <typename TA, typename TX, typename TY>
struct Axpy {
  const TA a;
  Axpy(const TA a, TX x, TY y): a(a) { ; }
  __device__ void operator()(TX &x, TY &y)
  {
    y += a * x;
  }
};
template <typename TA, typename TX, typename TY>
auto makeAxpy(const TA a, TX x, TY y) { return Axpy<TA,TX,TY>(a,x,y); }

template <typename TA, typename TX, typename TY>
void axpy(qudaStream_t s, TA a, TX x, TY y, const int n)
{
  auto f = makeAxpy(a, x[0], y[0]);
  auto arg = makeArg(x, y, f, n);
  auto nthreads = 32;
  qudaLaunch(2, nthreads, 0, s, Kernel, arg);
}

int main() {
  qudaDeviceProp prop;
  int dev = 0;
  qudaGetDeviceProperties(&prop, dev);
  std::cout << "Device : " << prop.name << std::endl;

  qudaStream_t s;
  qudaStreamCreate(&s);

  int n = 10000;
  auto x = (float *)qudaMallocManaged(n*sizeof(float));
  auto y = (float *)qudaMallocManaged(n*sizeof(float));
  for(int i=0; i<n; i++) {
    x[i] = i + 1;
    y[i] = 2*i + 1;
  }

  std::cout << "Inputs x: "<<x[0]<<", "<<x[1]<<", ..., "<<x[n-1]<<std::endl;
  std::cout << "Inputs y: "<<y[0]<<", "<<y[1]<<", ..., "<<y[n-1]<<std::endl;

  axpy(s, 2.0, x, y, n);

  qudaStreamSynchronize(s);
  std::cout << "Output: "<<y[0]<<", "<<y[1]<<", ..., "<<y[n-1]<<std::endl;
  for(int i=0; i<n; i++) {
    auto r = 4*i + 3;
    if(y[i]!=r) {
      std::cout << i << ": " << y[i] << std::endl;
    }
  }
}
