#include <vector>
#include "reduce_cuda.h"

template <typename T> struct plus {
  __device__ __host__ T operator()(T a, T b) { return a + b; }
};

template <typename T> struct maximum {
  __device__ __host__ T operator()(T a, T b) { return a > b ? a : b; }
};

template <typename T> struct minimum {
  __device__ __host__ T operator()(T a, T b) { return a < b ? a : b; }
};

template <typename T> struct identity {
  __device__ __host__ T operator()(T a) { return a; }
};

template <typename reduce_t, typename T, typename count_t,
	  typename transformer, typename reducer>
struct TransformReduceArg : public ReduceArg<reduce_t> {
  static constexpr int block_size = DEVPARAM_NTHREAD;
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

template <typename Arg>
__launch_bounds__(Arg::block_size) __global__ void
transform_reduce_kernel(Arg arg)
{
  using count_t = decltype(arg.n_items);
  using reduce_t = decltype(arg.init);

  count_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y;
  auto v = arg.v[j];
  reduce_t r_ = arg.init;

  while (i < arg.n_items) {
    auto v_ = arg.h(v[i]);
    r_ = arg.r(r_, v_);
    i += blockDim.x * gridDim.x;
  }

  reduce<Arg::block_size, reduce_t, false, decltype(arg.r)>(arg, r_, j);
}

template <typename Arg>
class TransformReduce
{
  Arg &arg;

public:
  TransformReduce(Arg &arg) : arg(arg) {}

  void apply(const cudaStream_t &stream)
  {
    uint gx = DEVPARAM_NTEAM;
    dim3 grid = { gx, (uint)arg.n_batch, 1 };
    dim3 block = { Arg::block_size, 1, 1 };
    if(1) {
      transform_reduce_kernel<<<grid, block>>>(arg);
      cudaDeviceSynchronize();
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
  TransformReduceArg<reduce_t, T, I, transformer, reducer>
    arg(v, n_items, h, init, r);
  TransformReduce<decltype(arg)> reduce(arg);
  reduce.apply(0);
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
double sum(cudaStream_t s, TX *x, const int n)
{
  double r = reduce<double>(x, n, 0.0, plus<TX>());
  return r;
}

#ifndef BENCHMARK
int main() {
  cudaError_t err;
  cudaDeviceProp prop;
  int dev = 0;
  err = cudaGetDeviceProperties(&prop, dev);
  std::cout << "Device : " << prop.name << std::endl;
  initReduce();

  cudaStream_t s;
  cudaStreamCreate(&s);

  int n = 1000;
  float *x;
  err = cudaMallocManaged(&x, n*sizeof(float));
  //printf("error: %i\n", err);
  for(int i=0; i<n; i++) {
    x[i] = i + 1;
  }

  std::cout << "Inputs x: "<<x[0]<<", "<<x[1]<<", ..., "<<x[n-1]<<std::endl;

  double r = sum(s, x, n);
  cudaStreamSynchronize(s);
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
#endif
