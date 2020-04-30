CXX = g++
CXXFLAGS = -g -O3 -std=c++17 -fopenmp
LDFLAGS = -fopenmp

DPCXX = dpcpp
DPCXXFLAGS = -g -O3 -std=c++17 -fsycl
DPLDFLAGS = -lOpenCL -lsycl

NVCXX = nvcc-10
NVCXXFLAGS = -g -O3 -std=c++14 -x cu
NVLDFLAGS =

#all: axpy_dpc
#all: axpy_cuda
#all: axpy_abs_cuda
#all: axpy_abs_dpc
#all: axpy_abs_omp
all: axpy_abs_cuda axpy_abs_dpc axpy_abs_omp

axpy_dpc: axpy_dpc.cpp
	$(DPCXX) $(DPCXXFLAGS) -o $@ $< $(DPLDFLAGS)

axpy_cuda: axpy_cuda.cpp
	$(NVCXX) $(NVCXXFLAGS) -o $@ $< $(NVLDFLAGS)

axpy_abs_cuda: axpy_abs.cpp backend_cuda.h
	$(NVCXX) $(NVCXXFLAGS) -o $@ $< $(NVLDFLAGS)

axpy_abs_dpc: axpy_abs.cpp backend_dpc.h
	$(DPCXX) $(DPCXXFLAGS) -o $@ $< $(DPLDFLAGS)

axpy_abs_omp: axpy_abs.cpp backend_omp.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean: 
	rm -rf axpy_dpc axpy_cuda axpy_abs_cuda axpy_abs_dpc axpy_abs_omp
