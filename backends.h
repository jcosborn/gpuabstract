#ifdef __NVCC__
#include "backend_cuda.h"
#elif defined(_OPENMP)
	#ifdef USE_OMP_TARGET
	#include "backend_omptarget.h"
	#else
	#include "backend_omp.h"
	#endif
#else
#include "backend_dpc.h"
#endif
