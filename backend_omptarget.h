#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <functional>

// xlC or g++ 10.1 does not recognize this, but this is the default?!
// #pragma omp requires unified_shared_memory

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
  int d = omp_get_default_device();
  snprintf(p->name, sizeof(p->name)-1, "device: %d", d);
}

void
qudaStreamSynchronize(qudaStream_t q)
{
#pragma omp taskwait
  ;
}

void *
qudaMallocManaged(size_t size)
{
  void *p = malloc(size);
  return p;
}

#define qudaLaunch(gridDim0, blockDim0, sharedMem0, stream0, func0, ...) \
do{	\
  int gd = gridDim0;	\
  int ld = blockDim0;	\
_Pragma("omp target teams num_teams(gd) nowait")	\
_Pragma("omp parallel num_threads(ld)")	\
  {	\
    gd = omp_get_num_teams();	\
    ld = omp_get_num_threads();	\
    int gi = omp_get_team_num();	\
    int li = omp_get_thread_num();	\
	\
    func0(__VA_ARGS__, li, ld, gi, gd);	\
  }	\
}while(0);
