#include "util.hpp"
#include "handle.hpp"


// (special) matrix multiplication: C = A * B
// A: m x k
// B: k x n
// C: m x n
void GEMM(int m, int n, int k, const double *A, const double *B, double *C) {
  auto const& handle = Handle_t::instance();
  double alpha = 1.0;
  double beta = 0.;
  CHECK_CUBLAS( cublasDgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N, 
        m, n, k, &alpha, A, m, B, k, &beta, C, m) );
}


