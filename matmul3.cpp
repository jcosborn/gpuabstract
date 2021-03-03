#include <iostream>
#include "backends.h"
#include "mat3.h"

template <typename TR, typename TX, typename TY>
struct MatmulArg {
  TR *r;
  TX *x;
  TY *y;
  dim1 threads;
  MatmulArg(TR *r, TX *x, TY *y, int length): r(r), x(x), y(y) {
    threads.x = length;
  }
};

template <typename Arg>
struct MatmulFunc {
  Arg arg;
  MatmulFunc(Arg &arg): arg(arg) {;}
  void operator() (int i) const {
    mul(arg.r[i], arg.x[i], arg.y[i]);
  }
};

template <typename TR, typename TX, typename TY>
class Matmul: Kern1d {
  TR *r;
  TX *x;
  TY *y;
  const int n;
public:
  Matmul(TR *r, TX *x, TY *y, const int n): r(r), x(x), y(y), n(n) {
    apply();
  }
  void apply(void) const
  {
    MatmulArg<TR,TX,TY> arg(r, x, y, n);
    launch<MatmulFunc>(arg);
  }
};

template <typename TR, typename TX, typename TY>
void matmul(TR *r, TX *x, TY *y, const int n)
{
  Matmul<TR,TX,TY>(r, x, y, n);
}

template <typename T>
void runtest(int n)
{
  using M = CMat3<T>;
  auto r = (M *)qudaMallocManaged(n*sizeof(M));
  auto x = (M *)qudaMallocManaged(n*sizeof(M));
  auto y = (M *)qudaMallocManaged(n*sizeof(M));
  for(int i=0; i<n; i++) {
    set(x[i], i+1);
    set(y[i], 2*i+1);
  }

  //std::cout << "Inputs x: "<<x[0]<<", "<<x[1]<<", ..., "<<x[n-1]<<std::endl;
  //std::cout << "Inputs y: "<<y[0]<<", "<<y[1]<<", ..., "<<y[n-1]<<std::endl;

  matmul(r, x, y, n);
  qudaStreamSynchronize();

  //std::cout << "Output: "<<y[0]<<", "<<y[1]<<", ..., "<<y[n-1]<<std::endl;
  for(int i=0; i<n; i++) {
    M t;
    mul(t, x[i], y[i]);
    t -= r[i];
    auto t2 = norm2(t);
    if(t2!=0.0) {
      std::cout << i << ": " << t2 << std::endl;
    }
  }
}

int main() {
  qudaInitDevice();

  int n = 10000;
  runtest<float>(n);
  runtest<double>(n);
}
