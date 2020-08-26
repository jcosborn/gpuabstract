#include"bench_sum.h"

#include"transform_reduce_omptarget.cpp"

int
main()
{
	struct {char name[256];} prop;
	int dev = omp_get_default_device();
	snprintf(prop.name, sizeof(prop.name)-1, "OMP device %d", dev);
	std::cout << "Device : " << prop.name << std::endl;
	initReduce();

	uint64_t nmax = 1<<30;
	BENCH_T *x = (BENCH_T *)malloc(nmax*sizeof(BENCH_T));

	BENCH_INIT;

    auto xd = to_device(x, nmax*sizeof(BENCH_T));

	BENCH_BEGIN;
	double r = sum(0, xd, n);
	BENCH_END;

    omp_target_free(xd, omp_get_default_device());
	free(x);
}
