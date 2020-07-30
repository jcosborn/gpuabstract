#pragma once
#include <omp.h>
#include <stdio.h>

// #pragma omp requires unified_shared_memory

template <typename T>
T * to_device(const T * x, size_t s) {
  const int d = omp_get_default_device();
  const int h = omp_get_initial_device();
  auto p = (T *)omp_target_alloc(s, d);
  printf("# to_device: host_ptr@%d = %p  device_ptr@%d = %p  size = %zu\n", h, x, d, p, s);
  omp_target_memcpy(p, (T *)x, s, 0, 0, d, h);
  return p;
}

template <typename T>
T * to_device(T& x) {
  constexpr size_t s = sizeof(T);
  return to_device(&x, s);
}

template <typename T>
void from_device(T * host_ptr, T * device_ptr, size_t s) {
  const int d = omp_get_default_device();
  const int h = omp_get_initial_device();
  printf("# from_device: host_ptr@%d = %p  device_ptr@%d = %p  size = %zu\n", h, host_ptr, d, device_ptr, s);
  omp_target_memcpy(host_ptr, device_ptr, s, 0, 0, h, d);
}

template <typename T>
void from_device(T * host_ptr, T * device_ptr) {
  constexpr size_t s = sizeof(T);
  from_device(host_ptr, device_ptr, s);
}

#define __global__
#define __device__
#define __host__
#define __shared__
#define __launch_bounds__(x)
typedef int cudaStream_t;

#define QUDA_MAX_MULTI_REDUCE 1024

__device__ __host__ inline void zero(double &v) { v = 0.0; }

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
unsigned int *getDeviceCountBuffer(void);
void fillHostReduceBufferFromDevice(void);

template <typename T>
struct ReduceArg {
  int n_batch;
  const T init;
  T *partial;
  T *result_d;
  T *result_h;
  unsigned int *count;
  ReduceArg(int n_batch = 1, T init = (T)0) :
    n_batch(n_batch),
    init(init),
    partial(static_cast<T*>(getDeviceReduceBuffer())),
    result_d(static_cast<T*>(getMappedHostReduceBuffer())),
    result_h(static_cast<T*>(getHostReduceBuffer())),
    count(static_cast<unsigned int*>(getDeviceCountBuffer()))
  {
    //  write reduction to GPU memory if asynchronous
    //if (commAsyncReduction()) result_d = partial;
  }
};

// naive implementation in omp
// https://github.com/NVlabs/cub/blob/1.8.0/cub/block/specializations/block_reduce_warp_reductions.cuh
// https://github.com/NVlabs/cub/blob/1.8.0/cub/warp/specializations/warp_reduce_smem.cuh
template <typename T, int block_dim_x, int block_dim_y = 1>
struct BlockReduce
{
  static constexpr int block_threads = block_dim_x*block_dim_y;
  static constexpr int warp_threads = 32;
  static constexpr int warp_steps = 5;  // log2(32)
  static constexpr int warps = (block_threads + warp_threads - 1) / warp_threads;
  struct TempStorage
  {
    T reduce[warps][warp_threads];
  };
  TempStorage &temp_storage;
  int tid,lane_id,warp_id;
  inline BlockReduce(TempStorage &temp_storage) :
    temp_storage(temp_storage),
    tid(omp_get_thread_num()),
    lane_id(tid%warp_threads),
    warp_id(tid/warp_threads)
  {}
  template <typename ReductionOp>
  inline T Reduce(T input, ReductionOp reduction_op)
  {
#pragma unroll
    for(int i=0;i<warp_steps;++i){
      temp_storage.reduce[warp_id][lane_id] = input;
      const int offset = 1 << i;
      // this is bad, but we don't have warp barrier
#pragma omp barrier
      input = reduction_op(input, temp_storage.reduce[warp_id][lane_id+offset]);
    }

#pragma omp barrier

    if (lane_id == 0) temp_storage.reduce[warp_id][0] = input;
#pragma omp barrier
    if (tid == 0) {
#pragma unroll
      for(int i=1;i<warps;++i) input = reduction_op(input, temp_storage.reduce[i][0]);
    }
    return input;
  }
};

// __device__ unsigned int count[QUDA_MAX_MULTI_REDUCE] = { };
// __shared__ bool isLastBlockDone;

template <int block_size_x, int block_size_y, typename T,
	  bool do_sum = true, typename Reducer = plus<T>>
__device__ inline void
reduce2d(ReduceArg<T> arg, const T &in, const int idx=0) {
  // assume block_size_x*block_size_y == num_threads
  typedef BlockReduce<T, block_size_x, block_size_y> BlockReduce;
  typename BlockReduce::TempStorage tmp;
#pragma omp allocate(tmp) allocator(omp_pteam_mem_alloc)

  Reducer r;
  // T aggregate = (do_sum ? BlockReduce(cub_tmp).Sum(in) :
  //                BlockReduce(cub_tmp).Reduce(in, r));
  T aggregate = BlockReduce(tmp).Reduce(in, r);

  bool isLastBlockDone;
#pragma omp allocate(isLastBlockDone) allocator(omp_pteam_mem_alloc)
  int gd = omp_get_num_teams();
  int ld = omp_get_num_threads();  // block_size_x*block_size_y
  int gi = omp_get_team_num();  // blockIdx.x + blockIdx.y*gridDim.x
  int li = omp_get_thread_num();  // threadIdx.x + threadIdx.y*blockDim.x
  auto nb = arg.n_batch;  // gridDim.y
  auto gdx = gd/nb;  // gridDim.x
  auto bix = gi%gdx;  // blockIdx.x
  auto biy = gi/gdx;  // blockIdx.y
  auto tix = li%block_size_x;  // threadIdx.x
  auto tiy = li/block_size_x;  // threadIdx.y
  if (li == 0) {
    arg.partial[idx*gdx + bix] = aggregate;
    // TODO: the flush may not be necessary because of the barrier down below
    // __threadfence(); // flush result

    // increment global block counter
    unsigned int value;
#pragma omp atomic capture
    value = arg.count[idx]++;  // if(arg.count[idx]>gdx)arg.count[idx]=0;

    // determine if last block
    isLastBlockDone = (value == (gdx-1));
  }

#pragma omp barrier
  // __syncthreads();

  // finish the reduction if last block
  if (isLastBlockDone) {
    unsigned int i = tiy*block_size_x + tix;
    T sum = arg.init;
    // zero(sum);  // This only works for sum.
    while (i<gdx) {
      sum = r(sum, arg.partial[idx*gdx + i]);
      //sum += arg.partial[idx*gdx + i];
      i += block_size_x*block_size_y;
    }

    // sum = (do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum,r));
    sum = BlockReduce(tmp).Reduce(sum,r);

    // write out the final reduced value
    if (li == 0) {
      arg.result_d[idx] = sum;
      arg.count[idx] = 0; // set to zero for next time
    }
  }
}

template <int block_size, typename T, bool do_sum = true,
	  typename Reducer = plus<T>>
__device__ inline void
reduce(ReduceArg<T> arg, const T &in, const int idx=0) {
  reduce2d<block_size, 1, T, do_sum, Reducer>(arg, in, idx);
}
