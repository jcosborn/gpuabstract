#include "reduce_cuda.h"

void *device_malloc(size_t size) {
  void *ptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    printf("Failed to allocate device memory of size %zu\n", size);
    exit(-1);
  }
  return ptr;
}

void *pinned_malloc(size_t size) {
  //void *ptr = aligned_malloc(a, size);
  void *ptr = malloc(size);
  cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
  if (err != cudaSuccess) {
    printf("Failed to register pinned memory of size %zu\n", size);
  }
  return ptr;
}

void *mapped_malloc(size_t size) {
  //void *ptr = aligned_malloc(a, size);
  void *ptr = malloc(size);
  cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (err != cudaSuccess) {
    printf("Failed to register host-mapped memory of size %zu\n", size);
  }
  return ptr;
}

static double *d_reduce=0;
static double *h_reduce=0;
static double *hd_reduce=0;

void initReduce(void) {
  size_t bytes = DEVPARAM_RESBUFLEN*8;
  d_reduce = (double *) device_malloc(DEVPARAM_NTEAM*8);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if(deviceProp.canMapHostMemory) {
    printf("using mapped memory\n");
    h_reduce = (double *) mapped_malloc(bytes);
    cudaHostGetDevicePointer(&hd_reduce, h_reduce, 0);
  } else {
    printf("using pinned memory\n");
    h_reduce = (double *) pinned_malloc(bytes);
    hd_reduce = d_reduce;
  }
}

double *getDeviceReduceBuffer(void) { return d_reduce; }
double *getMappedHostReduceBuffer(void) { return hd_reduce; }
double *getHostReduceBuffer(void) { return h_reduce; }
