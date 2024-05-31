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
#define CUBLAS_CHECK(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CUBLAS assert: %s %s %d\n", cudaGetErrorEnum(code), file, line);
      if (abort) exit(code);
   }
}

#include <cusolverDn.h>

static const char *cusolverError(cusolverStatus_t error) {
  switch (error) {
    case CUSOLVER_STATUS_SUCCESS:
      return "CUSOLVER_STATUS_SUCCESS";
    
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
  
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    
    case CUSOLVER_STATUS_INVALID_WORKSPACE:
      return "CUSOLVER_STATUS_INVALID_WORKSPACE";
    
    case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
      return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
    
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
  }
  return "<unknown>";
}

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t code = (err);                                                             \
        if (code != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %s at %s: %d\n", cusolverError(code), __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)



#endif

