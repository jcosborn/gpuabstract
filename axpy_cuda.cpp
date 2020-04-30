#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

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

template <typename Arg>
__global__ void Kernel(Arg arg)
{
  unsigned int i = blockIdx.x * (blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x * blockDim.x;
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
void axpy(cudaStream_t s, TA a, TX x, TY y, const int n)
{
  auto f = makeAxpy(a, x[0], y[0]);
  auto arg = makeArg(x, y, f, n);
  auto nthreads = 32;
  Kernel<<<1,nthreads,0,s>>>(arg);
}

int main() {
  cudaError_t err;
  cudaDeviceProp prop;
  int dev = 0;
  err = cudaGetDeviceProperties(&prop, dev);
  std::cout << "Device : " << prop.name << std::endl;

  cudaStream_t s;
  cudaStreamCreate(&s);

  int n = 10000;
  float *x, *y;
  err = cudaMallocManaged(&x, n*sizeof(float));
  //printf("error: %i\n", err);
  err = cudaMallocManaged(&y, n*sizeof(float));
  //printf("error: %i\n", err);
  for(int i=0; i<n; i++) {
    x[i] = i + 1;
    y[i] = 2*i + 1;
  }

  std::cout << "Inputs x: "<<x[0]<<", "<<x[1]<<", ..., "<<x[n-1]<<std::endl;
  std::cout << "Inputs y: "<<y[0]<<", "<<y[1]<<", ..., "<<y[n-1]<<std::endl;

  axpy(s, 2.0, x, y, n);

  cudaStreamSynchronize(s);
  std::cout << "Output: "<<y[0]<<", "<<y[1]<<", ..., "<<y[n-1]<<std::endl;
  for(int i=0; i<n; i++) {
    auto r = 4*i + 3;
    if(y[i]!=r) {
      std::cout << i << ": " << y[i] << std::endl;
    }
  }
}
