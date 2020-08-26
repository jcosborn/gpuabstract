#pragma once
#include<chrono>
#include<cmath>

#ifndef BENCH_T
#define BENCH_T double
#endif

#ifndef DEVPARAM_RESBUFLEN
#define DEVPARAM_RESBUFLEN -1
#endif
#ifndef DEVPARAM_WARP_THREADS
#define DEVPARAM_WARP_THREADS -1
#endif

struct Mstat
{
	double min;
	double mean;
	double max;
	uint64_t count;
	Mstat() :
		min(std::numeric_limits<double>::max()),
		mean(0.0),
		max(std::numeric_limits<double>::lowest()),
		count(0)
		{}
	void push(double x, uint64_t n)
	{
		double dn = (double)n;
		double xm = x/dn;
		if(xm<min) min=xm;
		if(xm>max) max=xm;
		count += n;
		mean += (xm-mean) * dn / (double)count;
	}
};

// quick and dirty macros, must conforms ids:
// x: the array
// nmax: maximum length
// n: array length for the current benchmark
// r: the result of sum, to be defined

// generate a low discrepancy sequence
#define BENCH_INIT do{ \
	const double c = 0.5 * (std::sqrt(5.0)-1.0); \
	double dtemp; \
	x[0] = c; \
	for(uint64_t i=1; i<nmax; i++) x[i] = (BENCH_T)std::modf(x[i-1]+c,&dtemp)-0.5; \
	std::cout << "Inputs x: "<<x[0]<<", "<<x[1]<<", ..., "<<x[nmax-1]<<std::endl; \
	}while(0)


#define BENCH_BEGIN \
	for(uint64_t n=1<<20; n<=nmax; n<<=1){ \
		double rc = 0.0; \
		_Pragma("omp parallel for simd reduction(+:rc)") \
		for(uint64_t i=0; i<n; i++) rc += x[i]; \
		const double nme = (double)n/1048576; \
		Mstat ms; \
		uint64_t nrep = 1; \
		double rs = 0.0; \
		uint64_t ntry = 16; \
		uint64_t firstN; \
		double firstT; \
		for(uint64_t j=0; j<ntry; j++){ \
			auto t0 = std::chrono::steady_clock::now(); \
			for(uint64_t i=0; i<nrep; i++){

#define BENCH_END \
				rs += r; \
			} \
			auto t = std::chrono::steady_clock::now(); \
			std::chrono::duration<double> dt = t-t0; \
			double mspm = 1e3*dt.count()/nme; \
			if(j>0){ \
				ms.push(mspm, nrep); \
				nrep = std::max(16ul, (uint64_t)(10.0/(nme*ms.min))); \
			}else{ \
				firstN = nrep; \
				firstT = mspm/(double)nrep; \
				nrep = std::max(16ul, (uint64_t)(10.0/(nme*firstT))); \
			} \
			/* std::cout << nrep <<std::endl; */ \
		} \
		printf("rbuf %d team %d thread %d warp %d size/MB %4ld  1st ms/ME %.6f  rest ms/ME %.6f %.6f %.6f  rep %ld\n", \
			DEVPARAM_RESBUFLEN, DEVPARAM_NTEAM, DEVPARAM_NTHREAD, DEVPARAM_WARP_THREADS, \
			n*sizeof(BENCH_T)/1048576, firstT, ms.min, ms.mean, ms.max, ms.count); \
	 \
		double r = rs/(double)(firstN+ms.count); \
		if(std::abs(rc-r)>1e-12){ \
			printf("     CPU sum: %.17g\n", rc); \
			printf("ERR: GPU sum: %.17g\n", r); \
		} \
	}
