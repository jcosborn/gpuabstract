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
  //auto q = qd;
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

template <typename T>
struct SharedMemAcc {
  using Acc = accessor<T,1,access::mode::read_write,access::target::local>;
  Acc a;
  SharedMemAcc(int n, handler &h): a(range<1>(n), h) {;}
  template <typename Dim, typename Fun, typename Arg, typename Mem0, typename... Mem>
  void launch_device(Dim ndi, Fun &f, const Arg &arg, Mem0 mem0, Mem... mem) const {
    T *p = a.get_pointer();
    mem0.launch_device(ndi, f, arg, mem..., p);
  }
};

template <typename T>
struct SharedMem {
  int n;
  SharedMem(int n): n(n) {;}
  template <typename Fun, typename Arg, typename Mem0, typename... Mem>
  void launch_device(handler &h, Fun &f, const Arg &arg, Mem0 mem0, Mem... mem) const {
    SharedMemAcc<T> m(n, h);
    mem0.launch_device(h, f, arg, mem..., m);
  }
};

struct Kern1dRun {
  template <typename Dim, typename Fun, typename Arg, typename... Mem>
  void launch_device(Dim ndi, Fun &f, const Arg &arg, Mem... mem) const {
    auto i = ndi.get_global_id(0);
    auto gridSize = ndi.get_global_range(0);
    while(i < arg.threads.x) {
      f(i, mem...);
      i += gridSize;
    }
  }
};

struct Kern1dArg {
  range<1> gr;
  range<1> lr;
  Kern1dArg(int ng, int nl): gr(ng), lr(nl) {;}
  template <typename Fun, typename Arg, typename Mem0, typename... Mem>
  void launch_device(handler &h, Fun &f, const Arg &arg, Mem0 mem0, Mem... mem) const {
    h.parallel_for<class launchDevice1d>
      (nd_range(gr,lr),
       [=](nd_item<1> ndi) {
	 Kern1dRun kr;
	 mem0.launch_device(ndi, f, arg, mem..., kr);
	   });
  }
};

struct Kern1d {
  template <template <typename> class Functor, typename Arg>
  void launch_host(const qudaStream_t &stream, const Arg &arg) const
  {
    Functor<Arg> f(const_cast<Arg &>(arg));
    for (int i = 0; i < (int)arg.threads.x; i++) {
      f(i);
    }
  }

  template <template <typename> class Functor, typename Arg>
  void launch_device(const qudaStream_t &stream, const Arg &arg) const
  {
    Functor<Arg> f(const_cast<Arg &>(arg));
    auto nl = 32;  // local
    auto ng = 2 * nl;  // global
    //auto r = nd_range<1>(range<1>(nglobal), range<1>(nlocal));
    auto s = const_cast<qudaStream_t &>(stream);
    //qudaLaunch(2, nthreads, 0, s, Kernel, f, arg);
    s.submit([&] (handler &h) {
	       h.parallel_for<class test>
		 (nd_range<1>(range<1>(ng), range<1>(nl)),
		  [=](nd_item<1> ndi) {
		    auto i = ndi.get_global_id(0);
		    auto gridSize = ndi.get_global_range(0);
		    while(i < arg.threads.x) {
		      f(i);
		      i += gridSize;
		    }
		  });
	     });
  }
  template <template <typename> class Functor, typename Arg, typename Mem0, typename... Mem>
  void launch_device(const qudaStream_t &stream, const Arg &arg, Mem0 mem0, Mem... mem) const
  {
    Functor<Arg> f(const_cast<Arg &>(arg));
    auto nl = 32;  // local
    auto ng = 2 * nl;  // global
    //auto r = nd_range<1>(range<1>(nglobal), range<1>(nlocal));
    auto s = const_cast<qudaStream_t &>(stream);
    //qudaLaunch(2, nthreads, 0, s, Kernel, f, arg);
    s.submit([&] (handler &h) {
	       Kern1dArg ka(ng, nl);
	       mem0.launch_device(h, f, arg, mem..., ka);
	     });
  }

  template <template <typename> class Functor, typename Arg, typename... Mem>
  void launch(const qudaStream_t &stream, const Arg &arg, Mem... mem) const
  {
    //launch_host<Functor, Arg>(stream, arg);
    launch_device<Functor, Arg>(stream, arg, mem...);
  }
};
