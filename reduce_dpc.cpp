#include "reduce_dpc.h"

int have_queue = 0;
queue qd;

queue
getQueue(void)
{
  if(have_queue==0) {
    have_queue = 1;
    //default_selector my_selector;
    host_selector my_selector;
    qd = queue(my_selector);
  }
  return qd;
}

void *device_malloc(size_t size) {
  default_selector my_selector;
  //auto q = queue(my_selector);
  auto q = getQueue();
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
  //auto q = queue(my_selector);
  auto q = getQueue();
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
  //auto q = queue(my_selector);
  auto q = getQueue();
  auto dev = q.get_device();
  auto ctx = q.get_context();
  void *ptr = sycl::malloc_shared(size, dev, ctx);
  if (ptr == nullptr) {
    printf("Failed to register host-mapped memory of size %zu\n", size);
  }
  return ptr;
}

static void *d_reduce=nullptr;
static void *h_reduce=nullptr;
static void *hd_reduce=nullptr;
static unsigned int *d_count=nullptr;

template <typename T>
void setZero(T *x, int n)
{
  queue q = getQueue();
  q.submit([&] (handler &h) { h.parallel_for<class setZero>
	(range<1>(n), [=](id<1> i) {
	  x[i] = 0;
	}); });
  q.wait();
}

void initReduce(void) {
  size_t bytes = 1024*8;
  d_reduce = (void *) device_malloc(bytes);
  h_reduce = (void *) pinned_malloc(bytes);
  hd_reduce = h_reduce;
  d_count = (unsigned int *) device_malloc(QUDA_MAX_MULTI_REDUCE*sizeof(unsigned int));
  setZero(d_count, QUDA_MAX_MULTI_REDUCE);
}

void *getDeviceReduceBuffer(void) { return d_reduce; }
void *getMappedHostReduceBuffer(void) { return hd_reduce; }
void *getHostReduceBuffer(void) { return h_reduce; }
unsigned int *getDeviceCountBuffer(void) { return d_count; }
