#include <vector>
#include "reduce_dpc.h"

queue qg;

template <typename T> struct plus {
  T operator()(T a, T b) { return a + b; }
};

template <typename T> struct maximum {
  T operator()(T a, T b) { return a > b ? a : b; }
};

template <typename T> struct minimum {
  T operator()(T a, T b) { return a < b ? a : b; }
};

template <typename T> struct identity {
  T operator()(T a) { return a; }
};

template <typename reduce_t, typename T, typename count_t,
	  typename transformer, typename reducer>
struct TransformReduceArg : public ReduceArg<reduce_t> {
  static constexpr int block_size = 256;
  static constexpr int n_batch_max = 4;
  const T *v[n_batch_max];
  count_t n_items;
  int n_batch;
  reduce_t init;
  reduce_t result[n_batch_max];
  transformer h;
  reducer r;
  TransformReduceArg(const std::vector<T *> &v, count_t n_items, transformer h, reduce_t init, reducer r) :
    n_items(n_items),
    n_batch(v.size()),
    init(init),
    h(h),
    r(r)
  {
    for (size_t j = 0; j < v.size(); j++) this->v[j] = v[j];
  }
};

template <typename Arg> void transform_reduce(Arg &arg)
{
  using count_t = decltype(arg.n_items);
  using reduce_t = decltype(arg.init);

  for (int j = 0; j < arg.n_batch; j++) {
    auto v = arg.v[j];
    reduce_t r_ = arg.init;
    for (count_t i = 0; i < arg.n_items; i++) {
      auto v_ = arg.h(v[i]);
      r_ = arg.r(r_, v_);
    }
    arg.result[j] = r_;
  }
}

template <typename Arg, typename Dims>
void transform_reduce_kernel(Arg arg, Dims ndi)
{
  using reduce_t = decltype(arg.init);

  auto i = ndi.get_global_id(0);
  auto j = ndi.get_group(1);
  auto v = arg.v[j];
  reduce_t r_ = arg.init;

  while (i < arg.n_items) {
    auto v_ = arg.h(v[i]);
    r_ = arg.r(r_, v_);
    i += ndi.get_global_range(0);
  }
  reduce<Arg::block_size, reduce_t, false, decltype(arg.r)>(arg, r_, j, ndi);
}

template <typename Arg>
class TransformReduce
{
  Arg &arg;

public:
  TransformReduce(Arg &arg) : arg(arg) {}

  void apply(queue q)
  {
    auto arg2 = arg;
    size_t gx = 1 * Arg::block_size;
    auto globalSize = range<2>(gx, arg.n_batch);
    auto localSize = range<2>(Arg::block_size, 1);
    if(1) {
      q.submit([&] (handler &h) { h.parallel_for<class transformReduce>
	    (nd_range<2>(globalSize, localSize),
	     [=](nd_item<2> ndi) {
	       transform_reduce_kernel(arg2, ndi);
	     }); });
      q.wait();
      printf("result_h[0]: %g\n", arg2.result_h[0]);
      for (decltype(arg.n_batch) j = 0; j < arg.n_batch; j++)
        arg.result[j] = arg.result_h[j];
    } else {
      transform_reduce(arg);
    }
  }

  //long long flops() const { return 0; } // just care about bandwidth
  //long long bytes() const { return arg.n_batch * arg.n_items * sizeof(*arg.v); }
};

template <typename reduce_t, typename T, typename I,
	  typename transformer, typename reducer>
void transform_reduce(std::vector<reduce_t> &result,
		      const std::vector<T *> &v, I n_items,
		      transformer h, reduce_t init, reducer r)
{
  if (result.size() != v.size()) {
    printf("result %lu and input %lu set sizes do not match", result.size(), v.size());
    exit(-1);
  }
  TransformReduceArg <reduce_t, T, I, transformer, reducer>
    arg(v, n_items, h, init, r);
  TransformReduce<decltype(arg)> reduce(arg);
  //reduce.apply(0);
  //default_selector my_selector;
  //queue q(my_selector);
  reduce.apply(qg);
  qg.wait();
  for (size_t j = 0; j < result.size(); j++) result[j] = arg.result[j];
}

template <typename reduce_t, typename T, typename I,
	  typename transformer, typename reducer>
reduce_t transform_reduce(const T *v, I n_items,
			  transformer h, reduce_t init, reducer r)
{
  std::vector<reduce_t> result = {0.0};
  std::vector<const T *> v_ = {v};
  transform_reduce(result, v_, n_items, h, init, r);
  return result[0];
}

template <typename reduce_t, typename T, typename I,
	  typename transformer, typename reducer>
void reduce(std::vector<reduce_t> &result, const std::vector<T *> &v,
	    I n_items, reduce_t init, reducer r)
{
  transform_reduce(result, v, n_items, identity<T>(), init, r);
}

template <typename reduce_t, typename T, typename I, typename reducer>
reduce_t reduce(const T *v, I n_items, reduce_t init, reducer r)
{
  std::vector<reduce_t> result = {0.0};
  std::vector<const T *> v_ = {v};
  transform_reduce(result, v_, n_items, identity<T>(), init, r);
  return result[0];
}

template <typename TX>
double sum(queue q, TX *x, const int n)
{
  double r = reduce<double>(x, n, 0.0, plus<double>());
  //double r = reduce<float>(x, n, 0.0f, plus<float>());
  return r;
}

int main() {
  default_selector my_selector;
  //queue q(my_selector);
  queue q = getQueue();
  qg = q;
  auto dev = q.get_device();
  std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl;
  initReduce();

  auto ctx = q.get_context();
  int n = 1000;
  auto *x = (float*)cl::sycl::malloc_shared(n*sizeof(float), dev, ctx);
  for(int i=0; i<n; i++) {
    x[i] = i + 1;
  }

  std::cout << "Inputs x: "<<x[0]<<", "<<x[1]<<", ..., "<<x[n-1]<<std::endl;

  double r = sum(q, x, n);
  q.wait();
  printf("r: %g\n", r);
  //printf("x[0]: %g\n", x[0]);

  double rc = 0.0;
  for(int i=0; i<n; i++) {
    rc += x[i];
  }
  if(rc!=r) {
    std::cout << r << " != " << rc << std::endl;
  }
}
