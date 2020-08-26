#include <cuda.h>
#include <cuda_runtime.h>
#include "cub/cub.cuh"

#ifndef DEVPARAM_NTEAM
#define DEVPARAM_NTEAM 32
#endif
#ifndef DEVPARAM_NTHREAD
#define DEVPARAM_NTHREAD 512
#endif
#ifndef DEVPARAM_RESBUFLEN
#define DEVPARAM_RESBUFLEN 1024
#endif

#define QUDA_MAX_MULTI_REDUCE DEVPARAM_RESBUFLEN

__device__ __host__ inline void zero(double &v) { v = 0.0; }


template <typename scalar, int n>
struct vector_type {
  scalar data[n];
  __device__ __host__ inline scalar& operator[](int i) { return data[i]; }
  __device__ __host__ inline const scalar& operator[](int i) const { return data[i]; }
  __device__ __host__ inline static constexpr int size() { return n; }
  __device__ __host__ inline void operator+=(const vector_type &a) {
#pragma unroll
    for (int i=0; i<n; i++) data[i] += a[i];
  }
  __device__ __host__ inline void operator=(const vector_type &a) {
#pragma unroll
    for (int i=0; i<n; i++) data[i] = a[i];
  }
  __device__ __host__ vector_type() {
#pragma unroll
    for (int i=0; i<n; i++) zero(data[i]);
  }
};

template<typename scalar, int n>
__device__ __host__ inline void zero(vector_type<scalar,n> &v) {
#pragma unroll
  for (int i=0; i<n; i++) zero(v.data[i]);
}

template<typename scalar, int n>
__device__ __host__ inline vector_type<scalar,n> operator+(const vector_type<scalar,n> &a, const vector_type<scalar,n> &b) {
  vector_type<scalar,n> c;
#pragma unroll
  for (int i=0; i<n; i++) c[i] = a[i] + b[i];
  return c;
}

void initReduce(void);
double *getDeviceReduceBuffer(void);
double *getMappedHostReduceBuffer(void);
double *getHostReduceBuffer(void);

template <typename T>
struct ReduceArg {
  T *partial;
  T *result_d;
  T *result_h;
  ReduceArg() :
    partial(static_cast<T*>(getDeviceReduceBuffer())),
    result_d(static_cast<T*>(getMappedHostReduceBuffer())),
    result_h(static_cast<T*>(getHostReduceBuffer()))
  {
    //  write reduction to GPU memory if asynchronous
    //if (commAsyncReduction()) result_d = partial;
  }
};

__device__ unsigned int count[QUDA_MAX_MULTI_REDUCE] = { };
__shared__ bool isLastBlockDone;

template <int block_size_x, int block_size_y, typename T,
	  bool do_sum=true, typename Reducer=cub::Sum>
__device__ inline void
reduce2d(ReduceArg<T> arg, const T &in, const int idx=0) {
  typedef cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
			   block_size_y> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_tmp;

  Reducer r;
  T aggregate = (do_sum ? BlockReduce(cub_tmp).Sum(in) :
		 BlockReduce(cub_tmp).Reduce(in, r));

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    arg.partial[idx*gridDim.x + blockIdx.x] = aggregate;
    __threadfence(); // flush result

    // increment global block counter
    unsigned int value = atomicInc(&count[idx], gridDim.x);

    // determine if last block
    isLastBlockDone = (value == (gridDim.x-1));
  }

  __syncthreads();

  // finish the reduction if last block
  if (isLastBlockDone) {
    unsigned int i = threadIdx.y*block_size_x + threadIdx.x;
    T sum;
    zero(sum);
    while (i<gridDim.x) {
      sum = r(sum, arg.partial[idx*gridDim.x + i]);
      //sum += arg.partial[idx*gridDim.x + i];
      i += block_size_x*block_size_y;
    }

    sum = (do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum,r));

    // write out the final reduced value
    if (threadIdx.y*block_size_x + threadIdx.x == 0) {
      arg.result_d[idx] = sum;
      count[idx] = 0; // set to zero for next time
    }
  }
}

template <int block_size, typename T, bool do_sum = true,
	  typename Reducer = cub::Sum>
__device__ inline void
reduce(ReduceArg<T> arg, const T &in, const int idx=0) {
  reduce2d<block_size, 1, T, do_sum, Reducer>(arg, in, idx);
}
