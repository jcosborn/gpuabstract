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
CXXTFLAGS = -g -O3 -std=c++14 -qsmp=omp -qoffload
LDTFLAGS = -qsmp=omp -qoffload
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
NVCXXFLAGS = -g -O3 -std=c++14 -x cu
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
	$(CXXT) -DUSE_OMP_TARGET $(CXXTFLAGS) -o $@ $< $(LDTFLAGS)


reduce_cuda.o: reduce_cuda.cpp reduce_cuda.h
	$(NVCXX) $(NVCXXFLAGS) -c -o $@ $< $(NVLDFLAGS)

transform_reduce_cuda.o: transform_reduce_cuda.cpp reduce_cuda.h
	$(NVCXX) $(NVCXXFLAGS) -c -o $@ $< $(NVLDFLAGS)

transform_reduce_cuda: transform_reduce_cuda.o reduce_cuda.o
	$(NVCXX) -o $@ $^ $(NVLDFLAGS)


reduce_dpc.o: reduce_dpc.cpp reduce_dpc.h
	$(DPCXX) $(DPCXXFLAGS) -c -o $@ $< $(NVLDFLAGS)

transform_reduce_dpc.o: transform_reduce_dpc.cpp reduce_dpc.h
	$(DPCXX) $(DPCXXFLAGS) -c -o $@ $< $(NVLDFLAGS)

transform_reduce_dpc: transform_reduce_dpc.o reduce_dpc.o
	$(DPCXX) -o $@ $^ $(NVLDFLAGS)


.PHONY: clean
clean: 
	rm -rf axpy_dpc axpy_cuda axpy_abs_cuda axpy_abs_dpc axpy_abs_omp
	rm -rf reduce_cuda.o transform_reduce_cuda.o transform_reduce_cuda
	rm -rf reduce_dpc.o transform_reduce_dpc.o transform_reduce_dpc
