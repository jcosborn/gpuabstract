#include"bench_sum.h"

#include"transform_reduce_omptarget.cpp"

int
main()
{
	struct {char name[256];} prop;
	int dev = omp_get_default_device();
	snprintf(prop.name, sizeof(prop.name)-1, "OMP device %d", dev);
	std::cout << "Device : " << prop.name << std::endl;

	uint64_t nmax = 1<<30;
	BENCH_T *x = (BENCH_T *)malloc(nmax*sizeof(BENCH_T));

	BENCH_INIT;

    auto xd = to_device(x, nmax*sizeof(BENCH_T));

	BENCH_BEGIN;
	double r = 0.0;
	#pragma omp target teams distribute parallel for simd reduction(+:r) num_teams(DEVPARAM_NTEAM) thread_limit(DEVPARAM_NTHREAD) is_device_ptr(xd)
	for(uint64_t m=0; m<n; ++m) r += xd[m];
	BENCH_END;

    omp_target_free(xd, omp_get_default_device());
	free(x);
}
