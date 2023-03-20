#include "rid.hpp"

#include <numeric> // std::iota


// compute column ID
void RandCPQR_column(const Mat &A, int rank, 
    std::vector<int>& sk, std::vector<int> &rd, Mat &T, int &flops) {

  Mat Z = RandRowSketch(A, rank);

  MKL_INT b = Z.rows();
  MKL_INT m = Z.cols();
  MKL_INT jpvt[ m ] = {0}; // all columnns are free
  MKL_INT info;
  double query[1], tau[ b ];

  // query the size of workspace 
  const MKL_INT negone = -1;
  dgeqp3( &b, &m, Z.data(), &b, jpvt, tau, query, &negone, &info );

  // compute factorization
  const MKL_INT lwork = query[0];
  double work[ lwork ];

  dgeqp3( &b, &m, Z.data(), &b, jpvt, tau, work, &lwork, &info );
  assert( info==0 );
  //delete[] work;

  for (int i=0; i<m; i++)
      jpvt[i]--;

  sk.resize(b);
  rd.resize(m-b);
  std::copy_n(jpvt, b, sk.begin());
  std::copy_n(jpvt+b, m-b, rd.begin());


  T = Z.leftCols(b).triangularView<Eigen::Upper>().
    solve( Z.rightCols(m-b) );

  int n = A.rows();
  flops = 2*m*n*b + 4*m*b*b/3.0 + b*b*(m-b);
}


void RandCPQR(const Mat &A, int rank, 
    std::vector<int>& sk, std::vector<int> &rd, Mat &T, int &flops) {

  Mat Y = RandColSketch(A, rank);
  Mat Z = Y.transpose();

  MKL_INT b = Z.rows();
  MKL_INT m = Z.cols();
  MKL_INT jpvt[ m ] = {0}; // all columnns are free
  MKL_INT info;
  double query[1], tau[ b ];

  // query the size of workspace 
  const MKL_INT negone = -1;
  dgeqp3( &b, &m, Z.data(), &b, jpvt, tau, query, &negone, &info );

  // compute factorization
  const MKL_INT lwork = query[0];
  double work[ lwork ];
  dgeqp3( &b, &m, Z.data(), &b, jpvt, tau, work, &lwork, &info );
  assert( info==0 );
  //delete[] work;

  for (int i=0; i<m; i++)
      jpvt[i]--;

  sk.resize(b);
  rd.resize(m-b);
  std::copy_n(jpvt, b, sk.begin());
  std::copy_n(jpvt+b, m-b, rd.begin());

  Mat W = Z.leftCols(b).triangularView<Eigen::Upper>().
    solve( Z.rightCols(m-b) );
  T = W.transpose();

  int n = A.cols();
  flops = 2*m*n*b + 4*m*b*b/3.0 + b*b*(m-b);
}

