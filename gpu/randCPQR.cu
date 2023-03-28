#include "rid.hpp"
#include "types.hpp"
#include "gemm.hpp"
#include "random.hpp"
#include "util.hpp"
#include "handle.hpp"
#include "print.hpp"
#include "timer.hpp"
#include "submatrix.hpp"

#include "magma.h"
#include "flops.hpp"

#include <numeric>      // std::iota


void RandCPQR_column(const double *A, int m, int n, int k,
    std::vector<int> &sk, std::vector<int> &rd, double *&T, double &flops) {

  Timer t;

  t.start();
  dvec Gmat(k*m);
  dvec Ymat(k*n);
  CHECK_CUDA( cudaMalloc((void **) &T,  sizeof(double)*k*(n-k) ));
  t.stop(); double t4 = t.elapsed_time();
  

  t.start();
  Random::Gaussian(Gmat, 0., 1.);
  t.stop(); double t0 = t.elapsed_time();
  
  
  t.start();
  double *G = thrust::raw_pointer_cast( Gmat.data() );
  double *Y = thrust::raw_pointer_cast( Ymat.data() );
  GEMM(k, n, m, G, A, Y);
  t.stop(); double t1 = t.elapsed_time();
  //print(Ymat, k, n, "Y");

  t.start();
  int info;
  int lwork = 2*n + ( n+1 ) * magma_get_dgeqp3_nb( k, n );
  int jpvt[n] = {0};

  dvec tau(k);
  dvec work(lwork);
  magmaDouble_ptr dtau  = thrust::raw_pointer_cast( tau.data() );
  magmaDouble_ptr dwork = thrust::raw_pointer_cast( work.data() );
  magma_dgeqp3_gpu(k, n, Y, k, jpvt, dtau, dwork, lwork, &info);
  assert( info==0 );
  t.stop(); double t2 = t.elapsed_time();

  //print(jpvt, n, "jpvt");
  //print(tau, "tau");

  
  t.start();
  double one = 1.0;
  double *R12 = Y + k*k;
  auto const& handle = Handle_t::instance();
  CHECK_CUBLAS( cublasDtrsm(handle.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, k, n-k, &one, Y, k, R12, k) );
  t.stop(); double t3 = t.elapsed_time();


  t.start();
  for (int i=0; i<n; i++)
    jpvt[i]--;
  sk.resize(k);
  rd.resize(n-k);
  std::copy_n(jpvt, k, sk.begin());
  std::copy_n(jpvt+k, n-k, rd.begin());
  thrust::copy_n( dptr(R12), k*(n-k), dptr(T) );
  t.stop(); double t5 = t.elapsed_time();


  flops = 2.*m*n*k + FLOPS_DGEQRF( k, n ) + 1.0*k*k*(n-k);


#if 0
  std::cout<<std::endl
    <<"--------------------\n"
    <<"  RandCPQR\n"
    <<"--------------------\n"
    <<"Alloc: "<<t4<<std::endl
    <<"Rand:  "<<t0<<std::endl
    <<"GEMM:  "<<t1<<std::endl
    <<"CPQR:  "<<t2<<std::endl
    <<"Solve: "<<t3<<std::endl
    <<"Copy:  "<<t5<<std::endl
    <<"--------------------\n"
    <<"Total: "<<t0+t1+t2+t3+t4+t5<<std::endl
    <<"--------------------\n"
    <<std::endl;
#endif
}

