#include"bench_sum.h"

#include"transform_reduce_cuda.cpp"

int
main()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "Device : " << prop.name << std::endl;
	initReduce();

	cudaStream_t s;
	cudaStreamCreate(&s);

	uint64_t nmax = 1<<30;
	BENCH_T *x;
	cudaMallocManaged(&x, nmax*sizeof(BENCH_T));

	BENCH_INIT;

	BENCH_BEGIN;
	double r = sum(s, x, n);
	cudaStreamSynchronize(s);
	BENCH_END;
}
