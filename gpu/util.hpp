#ifndef UTIL_GPU_HPP
#define UTIL_GPU_HPP

#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>      // cusparse
#include "cublas_v2.h"

//#include <algorithm>
//#include <numeric>
#include <stdio.h>         // printf
#include <stdlib.h>        // EXIT_FAILURE
#include <assert.h>
#include <cstring>


#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
//#include <thrust/reduce.h>
//#include <thrust/functional.h>
//#include <thrust/binary_search.h>

template <typename T>
using tvec = thrust::device_vector<T>;
  
typedef tvec<int> ivec;
typedef tvec<double> dvec;


template <typename T>
using tptr = thrust::device_ptr<T>; 

typedef tptr<int> iptr;
typedef tptr<double> dptr;


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        assert(false);                                                         \
    }                                                                          \
}


#ifndef FRONTERA
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
	printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        assert(false);                                                         \
    }                                                                          \
}
#else
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        assert(false);                                                         \
    }                                                                          \
}
#endif


static const char *cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

#define CHECK_CUBLAS(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CUBLAS assert: %s %s %d\n", cudaGetErrorEnum(code), file, line);
      if (abort) exit(code);
   }
}

#include <cusolverDn.h>

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)


template <typename T>
void print(const thrust::device_vector<T>& vec, const std::string &name) {
  std::cout<<std::endl<<name<<":"<<std::endl;
  for (int i=0; i<vec.size(); i++)
    std::cout<<vec[i]<<" ";
  std::cout<<std::endl<<std::endl;
}


template <typename T>
void print(const thrust::device_vector<T>& vec, int m, int n, const std::string &name) {
  // print matrix in column ordering
  std::cout<<std::endl<<name<<":"<<std::endl;
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++)
      std::cout<<vec[i+j*m]<<" ";
    std::cout<<std::endl;
  }
}


#endif

