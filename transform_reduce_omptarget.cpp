#include <vector>
#include <iostream>
#include "reduce_omptarget.h"

template <typename reduce_t, typename T, typename count_t,
	  typename transformer, typename reducer>
struct TransformReduceArg : public ReduceArg<reduce_t> {
  static constexpr int block_size = 512;
  static constexpr int n_batch_max = 4;
  const T *v[n_batch_max];
  count_t n_items;
  reduce_t init;
  reduce_t result[n_batch_max];
  transformer h;
  reducer r;
  TransformReduceArg(const std::vector<T *> &v, count_t n_items, transformer h, reduce_t init, reducer r) :
    ReduceArg<reduce_t>(v.size(), init),
    n_items(n_items),
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
transform_reduce_kernel(Arg& arg)
{
  using count_t = decltype(arg.n_items);
  using reduce_t = decltype(arg.init);

  int gd = omp_get_num_teams();
  int ld = omp_get_num_threads();  // blockDim.x
  int gi = omp_get_team_num();
  int li = omp_get_thread_num();  // threadIdx.x
  auto nb = arg.n_batch;
  auto gdx = gd/nb;  // gridDim.x
  auto bix = gi%gdx;  // blockIdx.x
  auto biy = gi/gdx;  // blockIdx.y
  count_t i = bix * ld + li;
  int j = biy;
  auto v = arg.v[j];
  reduce_t r_ = arg.init;
  while (i < arg.n_items) {
    auto v_ = arg.h(v[i]);
    r_ = arg.r(r_, v_);
    i += ld * gdx;
  }
  reduce<Arg::block_size, reduce_t, false, decltype(arg.r)>(arg, r_, j);
}

template <typename Arg>
class TransformReduce
{
  Arg& arg;

public:
  TransformReduce(Arg& arg) : arg(arg) {}

  void apply(const cudaStream_t &stream)
  {
    uint gx = 32;
    int gd = gx*arg.n_batch;
    int ld = Arg::block_size;
    if(1) {
      printf("launch transform_reduce_kernel %d %d\n",gd,ld);
      auto device_arg = to_device(arg);
      int nteams = 0;
      int nthreads = 0;
#pragma omp target teams num_teams(gd) is_device_ptr(device_arg) map(tofrom:nteams,nthreads)
{
      // shared local storage, workaround for the buggy support of allocator(omp_pteam_mem_alloc)
      typedef BlockReduce<decltype(arg.init), Arg::block_size, 1> BlockReduce;
      typename BlockReduce::TempStorage reduce_tmp;
      device_arg->reduce_tmp = (void*)&reduce_tmp;
      bool isLastBlockDone = false;
      device_arg->isLastBlockDone = &isLastBlockDone;
#pragma omp parallel num_threads(ld)
{
      if(omp_get_team_num()==0 && omp_get_thread_num()==0) {
        nteams = omp_get_num_teams();
        nthreads = omp_get_num_threads();
      }
      transform_reduce_kernel(*device_arg);
}
}
      printf("actual launched teams %d, threads %d\n", nteams, nthreads);
      omp_target_free(device_arg, omp_get_default_device());
/*
      double *partial = (double *)malloc(1024*8);
      from_device(partial, arg.partial, 1024*8);
      for (int j = 0; j < 100; j++) printf("arg.partial[%d] = %.16g\n", j, partial[j]);
      free(partial);
*/
      fillHostReduceBufferFromDevice();
      // printf("arg.result_d = %p\n", arg.result_d);
      // printf("arg.result_h = %p\n", arg.result_h);
      for (decltype(arg.n_batch) j = 0; j < arg.n_batch; j++) {
        // printf("arg.result_h[%d] = %.16g\n", j, arg.result_h[j]);
        arg.result[j] = arg.result_h[j];
      }
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

int main() {
  struct {char name[256];} prop;
  int dev = omp_get_default_device();
  snprintf(prop.name, sizeof(prop.name)-1, "OMP device %d", dev);
  std::cout << "Device : " << prop.name << std::endl;
  initReduce();

  int n = 1000;
  float *x = (float *)malloc(n*sizeof(float));
  for(int i=0; i<n; i++) {
    x[i] = i + 1;
  }

  std::cout << "Inputs x: "<<x[0]<<", "<<x[1]<<", ..., "<<x[n-1]<<std::endl;

  double r = 0.0;
  if(0) {
    #pragma omp target data map(tofrom:x[0:n]) use_device_ptr(x)
    r = sum(0, x, n);
  } else {
    auto xp = to_device(x, n*sizeof(float));
    r = sum(0, xp, n);
    omp_target_free(xp, omp_get_default_device());
  }

  printf("r: %.16g\n", r);
  //printf("x[0]: %g\n", x[0]);

  double rc = 0.0;
  for(int i=0; i<n; i++) {
    rc += x[i];
  }
  if(rc!=r) {
    std::cout << r << " != " << rc << std::endl;
  }
}
