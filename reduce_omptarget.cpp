#include "reduce_omptarget.h"

void *device_malloc(size_t size) {
  int d = omp_get_default_device();
  void *ptr = omp_target_alloc(size, d);
  if (NULL == ptr) {
    printf("Failed to allocate device memory of size %zu\n", size);
    exit(-1);
  }
  return ptr;
}

void *pinned_malloc(size_t size) {
  //void *ptr = aligned_alloc(a, size);
  void *ptr = malloc(size);
  // TODO: use omp to associate a device pointer
  //cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
  if (NULL == ptr) {
    printf("Failed to register pinned memory of size %zu\n", size);
    exit(-1);
  }
  return ptr;
}

static double *d_reduce=0;
static double *h_reduce=0;
static double *hd_reduce=0;
static unsigned int *d_count=0;

void initReduce(void) {
  size_t bytes = 1024*8;
  d_reduce = (double *) device_malloc(bytes);
  h_reduce = (double *) pinned_malloc(bytes);
  hd_reduce = h_reduce;
  d_count = (unsigned int *) device_malloc(QUDA_MAX_MULTI_REDUCE*sizeof(unsigned int));
#pragma omp target teams distribute parallel for is_device_ptr(d_count)
  for(int i=0;i<QUDA_MAX_MULTI_REDUCE;++i) d_count[i] = 0;
}

double *getDeviceReduceBuffer(void) { return d_reduce; }
double *getMappedHostReduceBuffer(void) { return hd_reduce; }
double *getHostReduceBuffer(void) { return h_reduce; }
unsigned int *getDeviceCountBuffer(void) { return d_count; }
