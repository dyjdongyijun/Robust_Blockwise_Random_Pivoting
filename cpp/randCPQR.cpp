#include "rid.hpp"
#include "timer.hpp"

#include <numeric> // std::iota
#include <iostream>


// compute column ID
void RandCPQR_column(const Mat &A, int rank, 
    std::vector<int>& sk, std::vector<int> &rd, Mat &T, int &flops, double &time) {

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

  Timer t; t.start();
  dgeqp3( &b, &m, Z.data(), &b, jpvt, tau, work, &lwork, &info );
  t.stop(); time = t.elapsed_time();
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
    std::vector<int>& sk, std::vector<int> &rd, Mat &T, int &flops, double &time) {

  time = 0.;

  Timer t; t.start();
  Mat Y = RandColSketch(A, rank);
  t.stop(); time += t.elapsed_time();
  
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

  t.start();
  dgeqp3( &b, &m, Z.data(), &b, jpvt, tau, work, &lwork, &info );
  t.stop(); time += t.elapsed_time();
  assert( info==0 );
  //delete[] work;

  for (int i=0; i<m; i++)
      jpvt[i]--;

  sk.resize(b);
  rd.resize(m-b);
  std::copy_n(jpvt, b, sk.begin());
  std::copy_n(jpvt+b, m-b, rd.begin());

  t.start();
  Mat W = Z.leftCols(b).triangularView<Eigen::Upper>().
    solve( Z.rightCols(m-b) );
  t.stop(); time += t.elapsed_time();
  
  T = W.transpose();

  int n = A.cols();
  flops = 2*m*n*b + 4*m*b*b/3.0 + b*b*(m-b);
}

