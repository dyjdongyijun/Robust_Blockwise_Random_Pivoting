#include "rid.hpp"
#include "util.hpp"
#include "timer.hpp"
#include "flops.hpp"
#include "print.hpp"

#include <iostream>
#include <numeric>
#include <assert.h>



void RandAdapLUPP(const Mat &A, 
    std::vector<int> &sk, std::vector<int> &rd, Mat &T, double &flops,
    double tol, int blk) {
  /*
   * Compute (row) ID decomposition
   */
  
  Timer t;

  MKL_INT m = A.rows();
  MKL_INT n = A.cols();

  assert( n > blk );

  // permutation matrix (1-based fortran style)
  std::vector<MKL_INT> P(m);
  std::iota( P.begin(), P.end(), 1 );

  t.start();
  int p = std::min(m, n);
  Mat L = Mat::Zero( m, p );
  t.stop(); double t6 = t.elapsed_time();


  t.start();
  Mat R = RandMat(A.cols(), blk, 0., 1./blk);
  t.stop(); double t0 = t.elapsed_time();


  t.start();
  L.leftCols(blk) = A*R;
  t.stop(); double t1 = t.elapsed_time();
  flops = 2.*m*n*blk;


  int nb = std::ceil( p/blk );
  assert( p%blk == 0 ); // for current implementation; will be removed
  
  double t2 = 0., t4 = 0.;

  MKL_INT k = 0; 
  const MKL_INT forwrd = 1;
  for (int i=0; i<nb; i++) {
    k = i*blk; // number of processed rows/columns
 

    t.start();
    MKL_INT a = m - k;
    MKL_INT b = i < nb-1 ? blk : p-(nb-1)*blk;
    MKL_INT ipiv[ b ]; // 1-based fortran style integers
    assert( int(b)==blk );
    
    MKL_INT info;
    MKL_INT ld = m;
    double *E = L.data() + k*m+k;
    dgetrf( &a, &b, E, &ld, ipiv, &info );
    assert( info==0 );
    flops = flops + FLOPS_DGETRF(a,b);
    t.stop(); t2 += t.elapsed_time();



    // global permutation (accumulation of local results)
    for (int j=0; j<b; j++) {
      MKL_INT tmp = P[ j+k ];
      P[ j+k ] = P[ ipiv[j]-1+k ];
      P[ ipiv[j]-1+k ] = tmp;
    }
    
    

    // local permutation of current iteration
    std::vector<MKL_INT> Phat(a);
    std::iota( Phat.begin(), Phat.end(), 1 );
    for (int j=0; j<b; j++) {
      MKL_INT tmp = Phat[ j ];
      Phat[ j ] = Phat[ ipiv[j]-1 ];
      Phat[ ipiv[j]-1 ] = tmp;
    }
    


    if (i>0) {
      t.start();
      dlapmr( &forwrd, &a, &k, L.data()+k, &m, Phat.data() );
      t.stop(); t4 += t.elapsed_time();
    }
    

    // next panel
    if (i == nb-1) break;
      
    b = std::min(blk, p-(i-1)*blk); // handle last panel
    k += blk; // number of processed rows/columns
    

    t.start();
    R = RandMat(A.cols(), b, 0., 1./b);
    t.stop(); t0 += t.elapsed_time();
    

    t.start();
    L.middleCols(k, b) = A * R;
    t.stop(); t1 += t.elapsed_time();
    flops = flops + 2.*m*n*b;
    

    t.start();
    dlapmr( &forwrd, &m, &b, L.data()+k*m, &m, P.data() );
    t.stop(); t4 += t.elapsed_time();


    t.start();
    L.block( k, k, m-k, b ) = L.block(k,k,m-k,b) - 
      L.bottomLeftCorner( m-k, k ) * 
      (L.topLeftCorner( k, k ).triangularView<Eigen::UnitLower>()
       .solve( L.block( 0, k, k, b ) ));
    t.stop(); t2 += t.elapsed_time();


    flops = flops + 1.*k*k*b + 2.*(m-k)*k*b;
    double eSchur = L.block(k,k,m-k,b).norm();
    //std::cout<<"Norm of Schur complement: "<<eSchur<<std::endl;
    if ( eSchur < tol ) break;
  }

  int r = k;
  sk.resize(r);
  rd.resize(m-r);


  for (int i=0; i<m; i++)
    P[i]--;
  auto *idx = P.data();
  std::copy_n(idx, r, sk.begin());
  std::copy_n(idx+r, m-r, rd.begin());
 

  t.start();
  T = L.topLeftCorner(r, r).triangularView<Eigen::UnitLower>().
    solve<Eigen::OnTheRight>( L.bottomLeftCorner(m-r, r) );
  t.stop(); double t3 = t.elapsed_time();


  flops = flops + 1.*r*r*(m-r);


#ifdef PROF
  std::cout<<std::endl
    <<"--------------------\n"
    <<"  RandAdapLUPP\n"
    <<"--------------------\n"
    <<"Aloc:  "<<t6<<std::endl
    <<"Rand:  "<<t0<<std::endl
    <<"GEMM:  "<<t1<<std::endl
    <<"LUPP:  "<<t2<<std::endl
    <<"Solve: "<<t3<<std::endl
    <<"Perm:  "<<t4<<std::endl
    <<"--------------------\n"
    <<"Total: "<<t0+t1+t2+t3+t4+t6<<std::endl
    <<"--------------------\n"
    <<std::endl;
#endif      
}



