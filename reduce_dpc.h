#include <CL/sycl.hpp>
using namespace cl::sycl;

#define QUDA_MAX_MULTI_REDUCE 1024

inline void zero(float &v) { v = 0.0; }
inline void zero(double &v) { v = 0.0; }


template <typename scalar, int n>
struct vector_type {
  scalar data[n];
    inline scalar& operator[](int i) { return data[i]; }
    inline const scalar& operator[](int i) const { return data[i]; }
    inline static constexpr int size() { return n; }
    inline void operator+=(const vector_type &a) {
#pragma unroll
    for (int i=0; i<n; i++) data[i] += a[i];
  }
    inline void operator=(const vector_type &a) {
#pragma unroll
    for (int i=0; i<n; i++) data[i] = a[i];
  }
    vector_type() {
#pragma unroll
    for (int i=0; i<n; i++) zero(data[i]);
  }
};

template<typename scalar, int n>
  inline void zero(vector_type<scalar,n> &v) {
#pragma unroll
  for (int i=0; i<n; i++) zero(v.data[i]);
}

template<typename scalar, int n>
inline vector_type<scalar,n>
operator+(const vector_type<scalar,n> &a, const vector_type<scalar,n> &b) {
  vector_type<scalar,n> c;
#pragma unroll
  for (int i=0; i<n; i++) c[i] = a[i] + b[i];
  return c;
}

void initReduce(void);
queue getQueue(void);
void *getDeviceReduceBuffer(void);
void *getMappedHostReduceBuffer(void);
void *getHostReduceBuffer(void);
unsigned int *getDeviceCountBuffer(void);

template <typename T>
struct ReduceArg {
  T *partial;
  T *result_d;
  T *result_h;
  unsigned int *count;
  ReduceArg() :
    partial(static_cast<T*>(getDeviceReduceBuffer())),
    result_d(static_cast<T*>(getMappedHostReduceBuffer())),
    result_h(static_cast<T*>(getHostReduceBuffer())),
    count(static_cast<unsigned int*>(getDeviceCountBuffer()))
  {
    //  write reduction to GPU memory if asynchronous
    //if (commAsyncReduction()) result_d = partial;
  }
};

// unsigned int count[QUDA_MAX_MULTI_REDUCE] = { };
//__shared__ bool isLastBlockDone;

template <int block_size_x, int block_size_y, typename T,
	  bool do_sum=true, typename Reducer=ONEAPI::plus<T>,
	  typename Dims>
inline void
reduce2d(ReduceArg<T> arg, const T &in, const int idx, Dims ndi) {
  auto grp = ndi.get_group();
  Reducer r;
  T aggregate = ONEAPI::reduce(grp, in, r);

  auto lid0 = ndi.get_local_id(0);
  auto lid1 = ndi.get_local_id(1);
  auto llid = ndi.get_local_linear_id();
  auto grpRg = ndi.get_group_range(0);
  auto grpId = ndi.get_group(0);
  bool isLastBlockDone = false;
  if (llid==0) {
    arg.partial[idx*grpRg + grpId] = aggregate;
    //__threadfence(); // flush result
    ndi.mem_fence(access::fence_space::global_and_local);

    // increment global block counter
    //unsigned int value = atomicInc(&count[idx], gridDim.x);
    atomic<unsigned int> acount { global_ptr<unsigned int> {&arg.count[idx]} };
    //auto gp = global_ptr<uint>(&arg.count[idx]);
    //auto acount = atomic<uint>(gp);
    auto value = acount.fetch_add(1);
    //value = acount.fetch_add(1);
    //arg.result_d[idx] = value + 1;
    //arg.result_d[idx] = arg.count[idx];

    // determine if last block
    isLastBlockDone = (value == (grpRg-1));
    //arg.result_d[idx] = isLastBlockDone;
  }
  isLastBlockDone = ONEAPI::broadcast(grp, isLastBlockDone);

  // finish the reduction if last block
  if (isLastBlockDone) {
    uint i = llid;
    T sum;
    zero(sum);
    while (i<grpRg) {
      sum = r(sum, arg.partial[idx*grpRg + i]);
      i += block_size_x*block_size_y;
    }

    sum = ONEAPI::reduce(grp, sum, r);

    // write out the final reduced value
    if (llid == 0) {
      arg.result_d[idx] = sum;
      arg.count[idx] = 0; // set to zero for next time
    }
  }
  //if (lid1*block_size_x + lid0 == 0) {
    //arg.result_d[idx] = aggregate;
    //arg.result_d[idx] = arg.partial[idx*grpRg + grpId];
    //arg.result_d[idx] = arg.count[idx];
    //arg.result_d[idx] = isLastBlockDone;
  //}
}

template <int block_size, typename T, bool do_sum = true,
	  typename Reducer=ONEAPI::plus<T>, typename Dims>
inline void
reduce(ReduceArg<T> arg, const T &in, const int idx, Dims ndi) {
  //reduce2d<block_size, 1, T, do_sum, Reducer>(arg, in, idx, ndi);
  using red = ONEAPI::plus<T>;
  reduce2d<block_size, 1, T, do_sum, red>(arg, in, idx, ndi);
}
