
EXE = driver
LIB = randid.so

OBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJS := $(filter-out driver.o, $(OBJS))
OBJS += $(patsubst %.cu,%.o,$(wildcard *.cu))


CFLAGS ?= -O3 -fopenmp -fPIC --std=c++14 -Wall -Wno-unused -DEIGEN_USE_MKL_ALL -DMKL_ILP64 -m64 
#CFLAGS ?= -g -fPIC --std=c++14 -Wall -Wno-unused -DEIGEN_USE_MKL_ALL -DMKL_ILP64 -m64 
VFLAGS ?= -O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC -Xcompiler -fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP


LFLAGS ?= -Wl,-rpath,$(shell pwd)
#LFLAGS += -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
LFLAGS += -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
LFLAGS += -lcublas -lcusolver -lcuda -lcudart 
LFLAGS += -ltbbmalloc_proxy 


INCLUDES = -I$(EIGEN_ROOT) -I$(EIGEN_RAND_ROOT) -I${MKLROOT}/include

all: $(EXE)

%.o: %.cpp %.hpp
	g++ -c $< -o $@ $(CFLAGS) $(INCLUDES)

%.o: %.cpp
	g++ -c $< -o $@ $(CFLAGS) $(INCLUDES)

%.o: %.cu
	nvcc -c $< -o $@ $(VFLAGS)

$(LIB): $(OBJS)
	g++ -shared $^ -o $@

driver: driver.o $(LIB)
	# TBB scalable memory allocation
	LD_PRELOAD=libtbbmalloc_proxy.so 
	g++ $^ -o $@ $(LFLAGS) 

clean:
	rm -f $(EXE) *.o $(LIB)

run: $(EXE)
	./$(EXE)
