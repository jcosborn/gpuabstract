#include <CL/sycl.hpp>
using namespace cl::sycl;

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

template <typename Arg, typename Dims>
void Kernel(Arg arg, Dims ndi)
{
  auto i = ndi.get_global_id(0);
  auto gridSize = ndi.get_global_range(0);
  while(i < arg.length) {
    arg.f(arg.x[i], arg.y[i]);
    i += gridSize;
  }
}

template <typename TA, typename TX, typename TY>
struct Axpy {
  const TA a;
  Axpy(const TA a, TX x, TY y): a(a) { ; }
  void operator()(TX &x, TY &y)
  {
    y += a * x;
  }
};

template <typename TA, typename TX, typename TY>
void axpy(queue q, TA a, TX x, TY y, const int n)
{
  auto f = Axpy(a, x[0], y[0]);
  auto arg = Arg(x, y, f, n);
  auto nthreads = 37;
  q.submit([&] (handler &h) { h.parallel_for<class axpy>
	(range<1>(nthreads),
	 //(nd_range<1>(range<1>(n), range<1>(2)),
	 [=](nd_item<1> ndi) {
	   Kernel(arg, ndi);
	 }); });
}

int main() {
  default_selector my_selector;
  queue q(my_selector);
  auto dev = q.get_device();
  std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl;

  auto ctx = q.get_context();
  int n = 1000;
  auto *x = (float*)cl::sycl::malloc_shared(n*sizeof(float), dev, ctx);
  auto *y = (float*)cl::sycl::malloc_shared(n*sizeof(float), dev, ctx);
  for(int i=0; i<n; i++) {
    x[i] = i + 1;
    y[i] = 2*i + 1;
  }

  std::cout << "Inputs x: "<<x[0]<<", "<<x[1]<<", ..., "<<x[n-1]<<std::endl;
  std::cout << "Inputs y: "<<y[0]<<", "<<y[1]<<", ..., "<<y[n-1]<<std::endl;

  axpy(q, 2.0, x, y, n);

  q.wait();
  std::cout << "Output: "<<y[0]<<", "<<y[1]<<", ..., "<<y[n-1]<<std::endl;
  for(int i=0; i<n; i++) {
    auto r = 4*i + 3;
    if(y[i]!=r) {
      std::cout << i << ": " << y[i] << std::endl;
    }
  }
}
