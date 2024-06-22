#include "id_cpqr.hpp"

#include "util.hpp"
#include "timer.hpp"
#include "types.hpp"
#include "flops.hpp"
#include "print.hpp"
#include "handle.hpp"
#include "submatrix.hpp"
#include "magma.h"


struct PermuteColumn : public thrust::unary_function<int, int> {
  iptr P;
  int  m;

  __host__
    PermuteColumn(iptr P_, int m_):
      P(P_), m(m_)  {}

  __device__
    int operator()(int i) {
      int col = i/m; // after permutation
      int row = i%m;
      return row + P[col] * m;
    }
};


void CPQR(int m, int n, double *A, 
    std::vector<int> ranks, std::vector<double> &error, std::vector<double> &t_ranks, 
    double &t_cpqr, double &flops) {
  
  //std::cout<<"Enter CPQR\n";

  dvec X(m*n);
  thrust::copy_n(dptr(A), m*n, X.begin());
  double Anrm2 = thrust::transform_reduce( 
      X.begin(), X.end(), thrust::square<double>(), 0., thrust::plus<double>());
  //std::cout<<"A norm squared: "<<Anrm2<<std::endl;
  //std::cout<<"finished initialization\n";
  //print(X, m, n, "X");

  Timer t; t.start();

  int info;
  int lwork = 2*n + ( n+1 ) * magma_get_dgeqp3_nb( m, n );
  int jpvt[n] = {0};
  //ivec jpvt(n, 0);

  dvec tau(std::min(m, n));
  dvec work(lwork);
  magmaDouble_ptr Xptr  = thrust::raw_pointer_cast( X.data() );
  magmaDouble_ptr dtau  = thrust::raw_pointer_cast( tau.data() );
  magmaDouble_ptr dwork = thrust::raw_pointer_cast( work.data() );
  //magma_int_t    *djpvt = thrust::raw_pointer_cast( jpvt.data() );
  magma_dgeqp3_gpu(m, n, Xptr, m, jpvt, dtau, dwork, lwork, &info);
  assert( info==0 );

  t.stop(); t_cpqr = t.elapsed_time();
  flops = FLOPS_DGEQRF(m, n);
  //std::cout<<"finished CPQR\n";
  //print(X, m, n, "QR");
  
  // construct A*P with permuted columns
  dvec AP(m*n);
  //thrust::copy_n(dptr(A), m*n, AP.begin());
  {
    for (int i=0; i<n; i++) jpvt[i]--;
    ivec djpvt(jpvt, jpvt+n);
    //print(djpvt, "pivots");
    
    auto zero = thrust::make_counting_iterator<int>(0);  
    auto iter = thrust::make_transform_iterator(zero, PermuteColumn(djpvt.data(), m));
    auto elem = thrust::make_permutation_iterator(dptr(A), iter);
    thrust::copy_n(elem, m*n, AP.begin());
  }
  //print(AP, m, n, "AP");
  

  // compute interpolation matrices
  // evaluate accuracies
  int k = ranks.size();
  t_ranks.resize(k);
  error.resize(k);
  
  // estimate the time for partial CPQR
#if 0 
  // a simple but under-estimator
  for (int i=0; i<k; i++)
    t_ranks[i] = t_cpqr / std::min(m,n) * ranks[i];
#else
  {
    int q = std::min(m,n);
    int p = std::max(m,n) - q;
    double a = 1./2*p*q*(q+1);
    double b = 1./6*q*(q+1)*(2*q+1);
    double total = a + b;
    for (int i=0; i<k; i++) {
      int r = q - ranks[i];
      t_ranks[i] = t_cpqr / total *
        (a-1./2*p*r*(r+1) + b-1./6*r*(r+1)*(2*r+1));
    }
  }
#endif

    
  for (int i=0; i<k; i++) {

    //std::cout<<"i: "<<i<<std::endl;
 
    // copy the A(1:r, :) for the interpolation matrix
    int r = ranks[i];
    assert(r > 0);
    dvec R(r*n);

    auto zero = thrust::make_counting_iterator<int>(0);  
    auto iter = thrust::make_transform_iterator(zero, SubMatrix(r, m));
    auto elem = thrust::make_permutation_iterator(X.begin(), iter);
    thrust::copy_n(elem, r*n, R.begin());
    //std::cout<<"Copy R matrix\n";
    //print(R, r, n, "R");


    t.start();
    double one = 1.0;
    double *R1 = thrust::raw_pointer_cast( R.data() );
    double *R2 = R1 + r*r;
    auto const& handle = Handle_t::instance();
    assert( r<std::min(m,n) );
    //std::cout<<"k: "<<k<<std::endl;
    
    CHECK_CUBLAS( 
        cublasDtrsm(handle.blas, 
          CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
          r, n-r, &one, R1, r, R2, r) );

    t.stop(); t_ranks[i] += t.elapsed_time();
    flops += 1.0*r*r*(n-r);
    //print(R, r, n, "T");
    //std::cout<<"finished interpolation matrix\n";
    //std::cout<<"m: "<<m<<", n: "<<n<<", r: "<<r<<std::endl;
    //std::cout<<m*(n-r)<<std::endl;

    // compute error E = Re - Sk * T
    // initialize to be the last n-r columns of AP
    dvec Emat(m*(n-r));
    thrust::copy_n(thrust::device, AP.begin()+r*m, m*(n-r), Emat.begin());
    //std::cout<<"init E matrix\n";
    
    double negone = -1.0;
    double *S = thrust::raw_pointer_cast( AP.data() );
    double *T = R2;
    double *E = thrust::raw_pointer_cast( Emat.data() );

    CHECK_CUBLAS(
        cublasDgemm(handle.blas, 
          CUBLAS_OP_N, CUBLAS_OP_N,
          m, n-r, r, 
          &negone, S, m,
          T, r,
          &one, E, m) );
    //std::cout<<"Error matrix\n";
    //print(Emat, m, n-r, "E");
    

    double Enrm2 = thrust::transform_reduce( 
        Emat.begin(), Emat.end(), thrust::square<double>(), 0., thrust::plus<double>());
    error[i] = Enrm2 / Anrm2;
    //std::cout<<"error: "<<error[i]<<std::endl;
  }
}

