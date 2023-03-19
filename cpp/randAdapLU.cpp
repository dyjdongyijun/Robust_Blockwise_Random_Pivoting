#include "rid.hpp"
#include "util.hpp"

#include <iostream>
#include <assert.h>


void RandAdapLUPP(const Mat &A, double tol, int blk) {
  /*
   * Compute (row) ID decomposition
   */
  
  int m = A.rows();
  int n = A.cols();

  assert( n > blk );

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P( m );
  P.setIdentity();

  int p = std::min(m, n);
  Mat L = Mat::Zero( m, p );
  Mat U = Mat::Zero( p, n );

  Mat Y = RandColSketch(A, blk);
  Mat E = Y;

  int nb = std::ceil( p/blk );
  assert( p%blk == 0 ); // for current implementation; will be removed
  
  for (int i=0; i<nb; i++) {
    int k = i*blk; // number of processed rows/columns
 
    MKL_INT b = E.cols(); // current block size
    MKL_INT ipiv[ b ]; // 1-based fortran style integers
    assert( int(b)==blk );
    
    MKL_INT a = E.rows();
    MKL_INT info;
    dgetrf( &a, &b, E.data(), &a, ipiv, &info );
    assert( info==0 );
    assert( E.rows() >= E.cols() );
    

    // U_1^t
    U.block(k, k, b, b) = E.topRows(b).triangularView<Eigen::Upper>();
    
    // global permutation (accumulation of local results)
    for (int j=0; j<b; j++)
      P.applyTranspositionOnTheLeft( k+j, k+ipiv[j]-1 );

    Mat L1 = E.topRows(b).triangularView<Eigen::UnitLower>();
    Mat L2 = E.bottomRows(a-b);

    // L matrix
    L.block(k, k, b, b) = L1;
    L.block(k+b, k, a-b, b) = L2;

    // local permutation of current iteration
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Phat( a );
    Phat.setIdentity();
    for (int j=0; j<b; j++)
      Phat.applyTranspositionOnTheLeft( j, ipiv[j]-1 );

    if (i>0) L.bottomLeftCorner( a, k ) = Phat * L.bottomLeftCorner( a, k );


    // next panel
    if (i == nb-1) break;
      
    b = std::min(blk, p-(nb-1)*blk); // handle last panel
    Y = P * RandColSketch(A, b);

    k += blk; // number of processed rows/columns
    U.block(0, k, k, b ) = 
      L.topLeftCorner( k,k ).triangularView<Eigen::Lower>()
      .solve( Y.topRows( k ) );

    E = Y.bottomRows( m-k ) - 
      L.bottomLeftCorner( m-k, k ) * U.block( 0, k, k, b );

    double eSchur = E.norm();
    std::cout<<"Norm of Schur complement: "<<eSchur<<std::endl;
    if ( eSchur < tol ) break;
  }
}


