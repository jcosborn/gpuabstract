#include "reduce_dpc.h"

void *device_malloc(size_t size) {
  default_selector my_selector;
  auto q = queue(my_selector);
  auto dev = q.get_device();
  auto ctx = q.get_context();
  void *ptr = sycl::malloc_device(size, dev, ctx);
  if (ptr == nullptr) {
    printf("Failed to allocate device memory of size %zu\n", size);
    exit(-1);
  }
  return ptr;
}

void *pinned_malloc(size_t size) {
  default_selector my_selector;
  auto q = queue(my_selector);
  auto dev = q.get_device();
  auto ctx = q.get_context();
  void *ptr = sycl::malloc_shared(size, dev, ctx);
  if (ptr == nullptr) {
    printf("Failed to register pinned memory of size %zu\n", size);
  }
  return ptr;
}

void *mapped_malloc(size_t size) {
  default_selector my_selector;
  auto q = queue(my_selector);
  auto dev = q.get_device();
  auto ctx = q.get_context();
  void *ptr = sycl::malloc_shared(size, dev, ctx);
  if (ptr == nullptr) {
    printf("Failed to register host-mapped memory of size %zu\n", size);
  }
  return ptr;
}

static double *d_reduce=nullptr;
static double *h_reduce=nullptr;
static double *hd_reduce=nullptr;
static unsigned int *d_count=nullptr;

void initReduce(void) {
  size_t bytes = 1024*8;
  d_reduce = (double *) device_malloc(bytes);
  h_reduce = (double *) pinned_malloc(bytes);
  hd_reduce = d_reduce;
  d_count = (unsigned int *) device_malloc(QUDA_MAX_MULTI_REDUCE*sizeof(unsigned int));
}

double *getDeviceReduceBuffer(void) { return d_reduce; }
double *getMappedHostReduceBuffer(void) { return hd_reduce; }
double *getHostReduceBuffer(void) { return h_reduce; }
unsigned int *getDeviceCountBuffer(void) { return d_count; }
