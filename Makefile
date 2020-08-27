NVARCH ?= sm_60

CXX ?= g++
ifeq ($(CXX),xlC)
CXXFLAGS = -g -O3 -std=c++14 -qsmp=omp
LDFLAGS = -qsmp=omp
else
CXXFLAGS = -g -O3 -std=c++17 -fopenmp
LDFLAGS = -fopenmp
endif

CXXT ?= g++
ifeq ($(CXXT),xlC)
CXXTFLAGS = -O3 -std=c++14 -qsmp=omp -qoffload
LDTFLAGS = -qsmp=omp -qoffload
else ifeq ($(CXXT),clang++)
CXXTFLAGS = -g -std=c++17 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=$(NVARCH)
LDTFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=$(NVARCH)
else ifeq ($(CXXT),icpx)
CXXTFLAGS = -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
LDTFLAGS = -fiopenmp -fopenmp-targets=spir64
else
CXXTFLAGS = -g -O3 -std=c++17 -fopenmp -foffload=nvptx-none -fno-stack-protector
LDTFLAGS = -fopenmp -foffload=nvptx-none -fno-stack-protector
endif

DPCXX = dpcpp
#DPCXXFLAGS = -g -O3 -std=c++17 -fsycl
#DPLDFLAGS = -lOpenCL -lsycl
DPCXXFLAGS = -O3 -std=c++17
DPLDFLAGS =

NVCXX = nvcc
NVCXXFLAGS = -O3 -std=c++14 -x cu
NVLDFLAGS =

#all: axpy_dpc
#all: axpy_cuda
#all: axpy_abs_cuda
#all: axpy_abs_dpc
#all: axpy_abs_omp
#all: axpy_abs_cuda axpy_abs_dpc axpy_abs_omp
#all: transform_reduce_cuda
all: transform_reduce_dpc

axpy_cuda: axpy_cuda.cpp
	$(NVCXX) $(NVCXXFLAGS) -o $@ $< $(NVLDFLAGS)

axpy_dpc: axpy_dpc.cpp
	$(DPCXX) $(DPCXXFLAGS) -o $@ $< $(DPLDFLAGS)

axpy_abs_cuda: axpy_abs.cpp backend_cuda.h
	$(NVCXX) $(NVCXXFLAGS) -o $@ $< $(NVLDFLAGS)

axpy_abs_dpc: axpy_abs.cpp backend_dpc.h
	$(DPCXX) $(DPCXXFLAGS) -o $@ $< $(DPLDFLAGS)

axpy_abs_omp: axpy_abs.cpp backend_omp.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

axpy_abs_omptarget: axpy_abs.cpp backend_omptarget.h
	$(CXXT) -DUSE_OMP_TARGET $(CXXTFLAGS) -o $@ $<


reduce_cuda.o: reduce_cuda.cpp reduce_cuda.h
	$(NVCXX) $(NVCXXFLAGS) -c -o $@ $< $(NVLDFLAGS)

transform_reduce_cuda.o: transform_reduce_cuda.cpp reduce_cuda.h
	$(NVCXX) $(NVCXXFLAGS) -c -o $@ $< $(NVLDFLAGS)

transform_reduce_cuda: transform_reduce_cuda.o reduce_cuda.o
	$(NVCXX) -o $@ $^ $(NVLDFLAGS)


reduce_omptarget.o: reduce_omptarget.cpp reduce_omptarget.h
	$(CXXT) $(CXXTFLAGS) -c -o $@ $<

transform_reduce_omptarget.o: transform_reduce_omptarget.cpp reduce_omptarget.h
	$(CXXT) $(CXXTFLAGS) -c -o $@ $<

transform_reduce_omptarget: transform_reduce_omptarget.o reduce_omptarget.o
	$(CXXT) -o $@ $^ $(LDTFLAGS)


reduce_dpc.o: reduce_dpc.cpp reduce_dpc.h
	$(DPCXX) $(DPCXXFLAGS) -c -o $@ $< $(NVLDFLAGS)

transform_reduce_dpc.o: transform_reduce_dpc.cpp reduce_dpc.h
	$(DPCXX) $(DPCXXFLAGS) -c -o $@ $< $(NVLDFLAGS)

transform_reduce_dpc: transform_reduce_dpc.o reduce_dpc.o
	$(DPCXX) -o $@ $^ $(NVLDFLAGS)


# Compile time parameters
CTP_NTEAMS=32 64 128 256 512 1024
CTP_NTHREADS=8 16 32 64 128 256 512 1024
CTP_RESBUFLENS=1 4 16 64 256 1024
CTP_OMPWARPS=1 2 4 8 16 32 64

define all_bench_sum_cuda =
BENCH_SUM_EXES += exe/bench_sum_cuda_rbuf$(3)_team$(1)_thread$(2)/exe
exe/bench_sum_cuda_rbuf$(3)_team$(1)_thread$(2)/exe: bench_sum.h reduce_cuda.h transform_reduce_cuda.cpp
exe/bench_sum_cuda_rbuf$(3)_team$(1)_thread$(2)/exe: bench_sum_cuda.cpp reduce_cuda.cpp
	mkdir -p exe/bench_sum_cuda_rbuf$(3)_team$(1)_thread$(2)
	cd exe/bench_sum_cuda_rbuf$(3)_team$(1)_thread$(2) && \
	$(NVCXX) $(NVCXXFLAGS) -Xcompiler=-fopenmp -o exe \
		-DBENCHMARK \
		-DDEVPARAM_NTEAM=$(1) \
		-DDEVPARAM_NTHREAD=$(2) \
		-DDEVPARAM_RESBUFLEN=$(3) \
		../../bench_sum_cuda.cpp ../../reduce_cuda.cpp
endef

define all_bench_sum_omptarget =
BENCH_SUM_EXES += exe/bench_sum_omptarget_rbuf$(3)_team$(1)_thread$(2)_warp$(4)/exe
exe/bench_sum_omptarget_rbuf$(3)_team$(1)_thread$(2)_warp$(4)/exe: bench_sum.h reduce_omptarget.h transform_reduce_omptarget.cpp
exe/bench_sum_omptarget_rbuf$(3)_team$(1)_thread$(2)_warp$(4)/exe: bench_sum_omptarget.cpp reduce_omptarget.cpp
	mkdir -p exe/bench_sum_omptarget_rbuf$(3)_team$(1)_thread$(2)_warp$(4)
	cd exe/bench_sum_omptarget_rbuf$(3)_team$(1)_thread$(2)_warp$(4) && \
	$(CXXT) $(CXXTFLAGS) -o exe \
		-DBENCHMARK \
		-DDEVPARAM_NTEAM=$(1) \
		-DDEVPARAM_NTHREAD=$(2) \
		-DDEVPARAM_RESBUFLEN=$(3) \
		-DDEVPARAM_WARP_THREADS=$(4) \
		../../bench_sum_omptarget.cpp ../../reduce_omptarget.cpp
endef

define all_bench_sum_omptarget_loop =
BENCH_SUM_EXES += exe/bench_sum_omptarget_loop_team$(1)_thread$(2)/exe
exe/bench_sum_omptarget_loop_team$(1)_thread$(2)/exe: bench_sum.h
exe/bench_sum_omptarget_loop_team$(1)_thread$(2)/exe: bench_sum_omptarget_loop.cpp
	mkdir -p exe/bench_sum_omptarget_loop_team$(1)_thread$(2)
	cd exe/bench_sum_omptarget_loop_team$(1)_thread$(2) && \
	$(CXXT) $(CXXTFLAGS) -o exe \
		-DBENCHMARK \
		-DDEVPARAM_NTEAM=$(1) \
		-DDEVPARAM_NTHREAD=$(2) \
		../../bench_sum_omptarget_loop.cpp
endef

$(foreach m,$(CTP_NTEAMS),$(foreach n,$(CTP_NTHREADS),$(foreach l,$(CTP_RESBUFLENS),$(eval $(call all_bench_sum_cuda,$(m),$(n),$(l))))))
$(foreach m,$(CTP_NTEAMS),$(foreach n,$(CTP_NTHREADS),$(foreach l,$(CTP_RESBUFLENS),$(foreach w,$(CTP_OMPWARPS),$(eval $(call all_bench_sum_omptarget,$(m),$(n),$(l),$(w)))))))
$(foreach m,$(CTP_NTEAMS),$(foreach n,$(CTP_NTHREADS),$(eval $(call all_bench_sum_omptarget_loop,$(m),$(n)))))

BENCHMARKS = $(filter-out %thread8_warp16/exe %thread8_warp32/exe %thread8_warp64/exe %thread16_warp32/exe %thread16_warp64/exe %thread32_warp64/exe,$(BENCH_SUM_EXES))
.PHONY: bench_sum
bench_sum: $(BENCHMARKS)

.PHONY: clean
clean: 
	rm -rf axpy_dpc axpy_cuda axpy_abs_cuda axpy_abs_dpc axpy_abs_omp
	rm -rf reduce_cuda.o transform_reduce_cuda.o transform_reduce_cuda
	rm -rf reduce_dpc.o transform_reduce_dpc.o transform_reduce_dpc
