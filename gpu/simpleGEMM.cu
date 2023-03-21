#include "util.hpp"

cublasHandle_t *handle = NULL;

void SimpleGEMM(int n, double *dA, double *dB, double *dC) {
  double alpha = 1.0, beta = 0.0;
  if (!handle) {
    handle = new cublasHandle_t;
    CHECK_CUBLAS( cublasCreate(handle) );
  }
  CHECK_CUBLAS( cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        n, n, n, &alpha, dA, n, dB, n, &beta, dC, n) );
}

