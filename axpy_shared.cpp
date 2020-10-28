#include <iostream>
#include "backends.h"

template <typename TA, typename TX, typename TY>
struct AxpyArg {
  const TA a;
  TX *x;
  TY *y;
  dim1 threads;
  AxpyArg(TA a, TX *x, TY *y, int length): a(a), x(x), y(y) {
    threads.x = length;
  }
};

template <typename Arg>
struct AxpyFunc {
  Arg arg;
  AxpyFunc(Arg &arg): arg(arg) {;}
  template <typename T>
  //void operator() (int i, T *m) const {
  void operator() (int i, T *m, T *m2) const {
    m[0] = arg.a;
    m2[4] = arg.a;
    arg.y[i] += arg.a * arg.x[i];
  }
};

template <typename TA, typename TX, typename TY>
class Axpy: Kern1d {
  const TA a;
  TX *x;
  TY *y;
  const int n;
public:
  Axpy(const TA a, TX *x, TY *y, const int n, qudaStream_t s): a(a), x(x), y(y), n(n) {
    apply(s);
  }
  void apply(qudaStream_t s) const
  {
    AxpyArg<TA,TX,TY> arg(a, x, y, n);
    SharedMem<TA> m(1);
    SharedMem<TA> m2(5);
    launch<AxpyFunc>(s, arg, m, m2);
  }
};

template <typename TA, typename TX, typename TY>
void axpy(qudaStream_t s, TA a, TX *x, TY *y, const int n)
{
  Axpy<TA,TX,TY>(a, x, y, n, s);
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

#ifndef DEVICE_NO_USM
    axpy(s, 2.0, x, y, n);
    qudaStreamSynchronize(s);
#else
  if(0) {	// if the compiler supports the pragma below
    #pragma omp target data map(to:x[0:n]) map(tofrom:y[0:n]) use_device_ptr(x,y)
    axpy(s, 2.0, x, y, n);
    qudaStreamSynchronize(s);
  } else {	// if the compiler does not support the above
    auto xp = (float *)omp_target_alloc(n*sizeof(float), dev);
    auto yp = (float *)omp_target_alloc(n*sizeof(float), dev);
    omp_target_memcpy(xp, x, n*sizeof(float), 0, 0, dev, omp_get_initial_device());
    omp_target_memcpy(yp, y, n*sizeof(float), 0, 0, dev, omp_get_initial_device());
    axpy(s, 2.0, xp, yp, n);
    qudaStreamSynchronize(s);
    omp_target_memcpy(y, yp, n*sizeof(float), 0, 0, omp_get_initial_device(), dev);
    omp_target_free(xp, omp_get_default_device());
    omp_target_free(yp, omp_get_default_device());
  }
#endif

  std::cout << "Output: "<<y[0]<<", "<<y[1]<<", ..., "<<y[n-1]<<std::endl;
  for(int i=0; i<n; i++) {
    auto r = 4*i + 3;
    if(y[i]!=r) {
      std::cout << i << ": " << y[i] << std::endl;
    }
  }
}
