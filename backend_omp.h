#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <functional>

#define __global__
#define __device__

#define KERN_ARGS int li, int ld, int gi, int gd

#if 0
template <typename T>
int getLocId(T ndi) { return ndi.get_local_id(0); }
template <typename T>
int getLocRange(T ndi) { return ndi.get_local_range(0); }
template <typename T>
int getGrpId(T ndi) { return ndi.get_group(0); }
template <typename T>
int getGrpRange(T ndi) { return ndi.get_group_range(0); }
#endif

int getGlobalId(KERN_ARGS)
{
  return (gi*ld) + li;
}

int getGlobalRange(KERN_ARGS)
{
  return gd * ld;
}

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

typedef int qudaStream_t;

void
qudaStreamCreate(qudaStream_t *q)
{
  *q = 0;
}

typedef struct {
  char name[256];
} qudaDeviceProp;

void
qudaGetDeviceProperties(qudaDeviceProp *p, int dev)
{
  auto name = "CPU OpenMP host device";
  strcpy(p->name, name);
}

void
qudaStreamSynchronize(qudaStream_t q)
{
  ;
}

void *
qudaMallocManaged(size_t size)
{
  void *p = malloc(size);
  return p;
}

void
qudaLaunch_(int gridDim0, int blockDim0, int sharedMem0, qudaStream_t stream0,
	    std::function<void(KERN_ARGS)> f)
{
  int gd = gridDim0;
  int ld = blockDim0;
#pragma omp parallel for collapse(2)
  for(int gi=0; gi<gd; gi++) {
    for(int li=0; li<ld; li++) {
      f(li, ld, gi, gd);
    }
  }
}

#define qudaLaunch(gridDim0, blockDim0, sharedMem0, stream0, func0, ...) \
  qudaLaunch_(gridDim0, blockDim0, sharedMem0, stream0,			\
	      [=](KERN_ARGS) {						\
		func0(__VA_ARGS__, li,ld,gi,gd);			\
	      });
