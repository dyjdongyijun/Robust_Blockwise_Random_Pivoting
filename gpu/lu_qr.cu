#include "util.hpp"
#include "types.hpp"
#include "handle.hpp"

#include "magma.h"


void LUPP(double *A, int n) {

  int m = n, k = n;
  double *Y = A;

  dvec work; // working memory for LU
  ivec ipiv(k); // local permutation
  ivec info(1); // an integer on gpu

  // query working space
  int lwork = 0;
  auto const& handle = Handle_t::instance();
  CUSOLVER_CHECK( cusolverDnDgetrf_bufferSize(handle.solver, m, k, Y, m, &lwork) );
  work.resize(lwork);
  
  // compute factorization
  CUSOLVER_CHECK( cusolverDnDgetrf(handle.solver, m, k, Y, m, 
        thrust::raw_pointer_cast(work.data()), 
        thrust::raw_pointer_cast(ipiv.data()), 
        thrust::raw_pointer_cast(info.data()) ));
  assert( info[0]==0 );
}


void CPQR(double *A, int n) {

  int k = n;
  double *Y = A;

  int info;
  int lwork = 2*n + ( n+1 ) * magma_get_dgeqp3_nb( k, n );
  int jpvt[n] = {0};

  dvec tau(k);
  dvec work(lwork);
  magmaDouble_ptr dtau  = thrust::raw_pointer_cast( tau.data() );
  magmaDouble_ptr dwork = thrust::raw_pointer_cast( work.data() );
  magma_dgeqp3_gpu(k, n, Y, k, jpvt, dtau, dwork, lwork, &info);
  assert( info==0 );
}


