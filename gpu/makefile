
EXE = driver
LIB = randid.so

OBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJS := $(filter-out driver.o, $(OBJS))
OBJS += $(patsubst %.cu,%.o,$(wildcard *.cu))


CFLAGS ?= -O3 -fPIC --std=c++14 -Wall -Wno-unused -DEIGEN_USE_MKL_ALL -DMKL_ILP64 -m64 
#CFLAGS ?= -g -fPIC --std=c++14 -Wall -Wno-unused -DEIGEN_USE_MKL_ALL -DMKL_ILP64 -m64 
VFLAGS ?= -O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC 


LFLAGS ?= -Wl,-rpath,$(shell pwd)
LFLAGS += -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
LFLAGS += -lcublas -lcusolver -lcuda -lcudart 

INCLUDES = -I$(EIGEN_ROOT) -I$(EIGEN_RAND_ROOT) -I${MKLROOT}/include

all: $(EXE)

%.o: %.cpp %.hpp
	g++ -c $< -o $@ $(CFLAGS) $(INCLUDES)

%.o: %.cpp
	g++ -c $< -o $@ $(CFLAGS) $(INCLUDES)

%.o: %.cu
	nvcc -c $< -o $@ $(VFLAGS)

randCPQR.o: randCPQR.cu rid.hpp
	nvcc -c $< -o $@ $(VFLAGS) -I$(MAGMADIR)/include -DADD_ -DMAGMA_WITH_MKL -DHAVE_CUBLAS

$(LIB): $(OBJS)
	g++ -shared $^ -o $@

driver: driver.o $(LIB)
	g++ $^ -o $@ $(LFLAGS) -L$(MAGMADIR)/lib -lmagma

clean:
	rm -f $(EXE) *.o $(LIB)

run: $(EXE)
	./$(EXE)
