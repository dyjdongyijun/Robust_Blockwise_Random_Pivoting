
EXE = driver
LIB = randid.so

OBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJS += $(patsubst %.cu,%.o,$(wildcard *.cu))
OBJS := $(filter-out driver.o bmk_lu_qr.o lu_qr.o, $(OBJS))
#OBJS := $(filter-out driver.o bmk_lu_qr.o lu_qr.o randCPQR.o, $(OBJS))

CXX = icpc -diag-disable=10441

CFLAGS ?= -O3 -fPIC -Wall -Wno-unused -DEIGEN_USE_MKL_ALL
#CFLAGS ?= -g -fPIC -Wall -Wno-unused -DEIGEN_USE_MKL_ALL
#VFLAGS ?= -arch=compute_70 -code=sm_70 -Xcompiler -fPIC 
VFLAGS ?= -DPROF -arch=compute_70 -code=sm_70 -Xcompiler -fPIC 
VFLAGS += -I$(MAGMA_ROOT)/include -DMAGMA_WITH_MKL -DHAVE_CUBLAS


LFLAGS ?= -Wl,-rpath,$(shell pwd)
#LFLAGS += -m64 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
LFLAGS += -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
LFLAGS += -lcublas -lcusolver -lcuda -lcudart 

INCLUDES = -I$(EIGEN_ROOT) -I${MKLROOT}/include
#INCLUDES = -I$(EIGEN_ROOT) -DMKL_ILP64 -m64 -I${MKLROOT}/include

all: $(EXE)

%.o: %.cpp %.hpp
	$(CXX) -c $< -o $@ $(CFLAGS) $(INCLUDES)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CFLAGS) $(INCLUDES)

%.o: %.cu
	nvcc -c $< -o $@ $(VFLAGS)

randCPQR.o: randCPQR.cu rid.hpp
	nvcc -c $< -o $@ $(VFLAGS) 

$(LIB): $(OBJS)
	$(CXX) -shared $^ -o $@

driver: driver.o $(LIB)
	$(CXX) $^ -o $@ $(LFLAGS) -L$(MAGMA_ROOT)/lib -lmagma

clean:
	rm -f $(EXE) *.o $(LIB)

run: $(EXE)
	./$(EXE)
