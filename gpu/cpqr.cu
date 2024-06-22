#include "cpqr.hpp"

#include "types.hpp"
#include "magma.h"


void CPQR(int m, int n, double *dA, int ld) {
  int info;
  int lwork = 2*n + ( n+1 ) * magma_get_dgeqp3_nb( m, n );
  int jpvt[n] = {0};

  dvec tau(std::min(m, n));
  dvec work(lwork);
  magmaDouble_ptr dtau  = thrust::raw_pointer_cast( tau.data() );
  magmaDouble_ptr dwork = thrust::raw_pointer_cast( work.data() );
  magma_dgeqp3_gpu(m, n, magmaDouble_ptr(dA), ld, jpvt, dtau, dwork, lwork, &info);
  assert( info==0 );

  for (int i=0; i<n; i++) jpvt[i]--;
}

