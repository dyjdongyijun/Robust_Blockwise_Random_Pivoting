#include "rid.hpp"
#include "types.hpp"
#include "gemm.hpp"
#include "random.hpp"
#include "util.hpp"
#include "handle.hpp"
#include "print.hpp"
#include "timer.hpp"
#include "submatrix.hpp"
#include "interp.hpp"

#include "magma.h"
#include "flops.hpp"

#include <numeric>      // std::iota



// This function is modified from RandCPQR
void RandCPQR_OS(const double *A, int m, int n, int r,
    int *h_sk, int *h_rd, double *&T, double &flops) {

  // over-sampling
  int k = 2*r;

  Timer t; t.start();
  dvec Gmat(k*m);
  dvec Ymat(k*n);
  if (!T)  CHECK_CUDA( cudaFree(T)  );
  CHECK_CUDA( cudaMalloc((void **) &T,  sizeof(double)*r*(n-r) ));  
  t.stop(); double t4 = t.elapsed_time();
  

  t.start();
  Random::Gaussian(Gmat, 0., 1./std::sqrt(m));
  t.stop(); double t0 = t.elapsed_time();
  
  
  t.start();
  double *G = thrust::raw_pointer_cast( Gmat.data() );
  double *Y = thrust::raw_pointer_cast( Ymat.data() );
  GEMM(k, n, m, G, A, Y); // Y = G * A
  t.stop(); double t1 = t.elapsed_time();
  //print(Ymat, k, n, "Y");

  // CPQR on Y(1:r, 1:n)
  t.start();
  std::vector<int> jpvt(n, 0);
  {
    // copy the first r rows
    dvec Yrmat(r*n); double *Yr = thrust::raw_pointer_cast( Yrmat.data() );
    auto ZERO = thrust::make_counting_iterator<int>(0);
    auto indx = thrust::make_transform_iterator(ZERO, SubMatrix(r, k));
    auto elem = thrust::make_permutation_iterator(dptr(Y), indx);
    thrust::copy_n( elem, r*n, Yrmat.begin() );   

    int info;
    int lwork = 2*n + ( n+1 ) * magma_get_dgeqp3_nb( r, n );

    dvec tau(k);
    dvec work(lwork);
    magmaDouble_ptr dtau  = thrust::raw_pointer_cast( tau.data() );
    magmaDouble_ptr dwork = thrust::raw_pointer_cast( work.data() );
    magma_dgeqp3_gpu(r, n, Yr, r, jpvt.data(), dtau, dwork, lwork, &info);
    assert( info==0 );
    
    // 0-based indexing in C++
    for (int i=0; i<n; i++)
      jpvt[i]--;

    //print(jpvt, "jpvt");
    //print(tau, "tau");
  }
  t.stop(); double t2 = t.elapsed_time();

  
  t.start(); 
  ivec P( jpvt.data(), jpvt.data()+n );
  t.stop(); double t5 = t.elapsed_time();



  t.start(); 
  std::copy_n( jpvt.data(), r, h_sk );
  std::copy_n( jpvt.data()+r, n-r, h_rd );
  

  Compute_interpolation_matrix(k, n, r, Ymat, P, T);
  t.stop(); double t3 = t.elapsed_time();


  flops = 2.*m*n*k + FLOPS_DGEQRF( k, n ) + 1.0*k*k*(n-k);


#ifdef PROF
  std::cout<<std::endl
    <<"--------------------\n"
    <<"  RandCPQR-OS\n"
    <<"--------------------\n"
    //<<"Alloc: "<<t4<<std::endl
    //<<"Rand:  "<<t0<<std::endl
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

