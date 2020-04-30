#include <CL/sycl.hpp>
using namespace cl::sycl;

#define __global__
#define __device__

template <typename T>
int getLocId(T ndi) { return ndi.get_local_id(0); }
template <typename T>
int getLocRange(T ndi) { return ndi.get_local_range(0); }
template <typename T>
int getGrpId(T ndi) { return ndi.get_group(0); }
template <typename T>
int getGrpRange(T ndi) { return ndi.get_group_range(0); }
template <typename T>
int getGlobalId(T ndi) { return ndi.get_global_id(0); }
template <typename T>
int getGlobalRange(T ndi) { return ndi.get_global_range(0); }

#define threadIdx_x getLocId(ndi...)
#define blockDim_x getLocRange(ndi...)
#define blockIdx_x getGrpId(ndi...)
#define gridDim_x getGrpRange(ndi...)
#define globalIdx_x getGlobalId(ndi...)
#define globalDim_x getGlobalRange(ndi...)

typedef struct {
  int x;
} dim1;

#define threadIdx ((dim1){threadIdx_x})
#define blockDim ((dim1){blockDim_x})
#define blockIdx ((dim1){blockIdx_x})
#define gridDim ((dim1){gridDim_x})

typedef queue qudaStream_t;

queue
getQueue(void)
{
  default_selector my_selector;
  return queue(my_selector);
}

void
qudaStreamCreate(queue *q)
{
  *q = getQueue();
}

typedef struct {
  char name[256];
} qudaDeviceProp;

void
qudaGetDeviceProperties(qudaDeviceProp *p, int dev)
{
  auto q = getQueue();
  auto d = q.get_device();
  auto name = d.get_info<info::device::name>();
  strcpy(p->name, name.c_str());
}

void
qudaStreamSynchronize(qudaStream_t q)
{
  q.wait();
}

void *
qudaMallocManaged(size_t size)
{
  auto q = getQueue();
  auto dev = q.get_device();
  auto ctx = q.get_context();
  void *p = cl::sycl::malloc_shared(size, dev, ctx);
  return p;
}

#define qudaLaunch(gridDim0, blockDim0, sharedMem0, stream0, func0, ...) \
  stream0.submit([&] (handler &h) { h.parallel_for<class test> \
	(nd_range<1>(range<1>(gridDim0*blockDim0), range<1>(blockDim0)), \
	 [=](nd_item<1> ndi) { \
	   func0(__VA_ARGS__, ndi); \
	 }); });
