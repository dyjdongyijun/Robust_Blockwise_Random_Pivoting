#include "rid.hpp"

#include <numeric> // std::iota
                   //

void RandLUPP(const Mat &A, int rank, 
    std::vector<int>& sk, std::vector<int> &rd, Mat &T, int &flops) {

  Mat Y = RandColSketch(A, rank);
  
  MKL_INT m = Y.rows();
  MKL_INT b = rank;
  MKL_INT info;
  MKL_INT ipiv[ b ];

  dgetrf( &m, &b, Y.data(), &m, ipiv, &info);
  assert( info==0 );

  std::vector<int> perm(m);
  std::iota(perm.begin(), perm.end(), 0);
  
  for (int i=0; i<b; i++) {
    int tmp = perm[ i ];
    perm[ i ] = perm[ ipiv[i]-1 ];
    perm[ ipiv[i]-1 ] = tmp;
  }

  sk.resize(b);
  rd.resize(m-b);
  std::copy_n(perm.begin(), b, sk.begin());
  std::copy_n(perm.begin()+b, m-b, rd.begin());

  T = Y.topRows(b).triangularView<Eigen::UnitLower>().
    solve<Eigen::OnTheRight>( Y.bottomRows(m-b) );

  int n = A.cols();
  flops = 2*m*n*b + 2*m*b*b/3.0 + b*b*(m-b);
}

