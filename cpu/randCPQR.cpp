#include "rid.hpp"
#include "timer.hpp"
#include "flops.hpp"

#include <numeric> // std::iota
#include <iostream> 


// compute column ID
void RandCPQR_column(const Mat &A, int rank, 
    std::vector<int>& sk, std::vector<int> &rd, Mat &T, double &flops) {

  Timer t;

  t.start();
  Mat R = RandMat(rank, A.rows(), 0., 1.);
  t.stop(); double t0 = t.elapsed_time();
  

  t.start();
  Mat Z = R*A;
  t.stop(); double t1 = t.elapsed_time();


  t.start();
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
  t.stop(); double t2 = t.elapsed_time();


  for (int i=0; i<m; i++)
      jpvt[i]--;

  sk.resize(b);
  rd.resize(m-b);
  std::copy_n(jpvt, b, sk.begin());
  std::copy_n(jpvt+b, m-b, rd.begin());


  t.start();
  T = Z.leftCols(b).triangularView<Eigen::Upper>().
    solve( Z.rightCols(m-b) );
  t.stop(); double t3 = t.elapsed_time();

  int n = A.rows();
  flops = 2.*m*n*b + FLOPS_DGEQRF(m,b) + 1.*b*b*(m-b);


#ifdef PROF
  std::cout<<std::endl
    <<"--------------------\n"
    <<"  RandCPQR\n"
    <<"--------------------\n"
    <<"Rand:  "<<t0<<std::endl
    <<"GEMM:  "<<t1<<std::endl
    <<"CPQR:  "<<t2<<std::endl
    <<"Solve: "<<t3<<std::endl
    <<"--------------------\n"
    <<"Total: "<<t0+t1+t2+t3<<std::endl
    <<"--------------------\n"
    <<std::endl;
#endif     
}


void RandCPQR(const Mat &A, int rank, 
    std::vector<int>& sk, std::vector<int> &rd, Mat &T, double &flops) {

  Timer t;

  t.start();
  Mat R = RandMat(A.cols(), rank, 0., 1.);
  t.stop(); double t0 = t.elapsed_time();


  t.start();
  Mat Y = A*R;
  t.stop(); double t1 = t.elapsed_time();

  
  t.start();
  Mat Z = Y.transpose();
  t.stop(); double t4 = t.elapsed_time();

  
  t.start();
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
  t.stop(); double t2 = t.elapsed_time();


  for (int i=0; i<m; i++)
      jpvt[i]--;

  sk.resize(b);
  rd.resize(m-b);
  std::copy_n(jpvt, b, sk.begin());
  std::copy_n(jpvt+b, m-b, rd.begin());

  t.start();
  Mat W = Z.leftCols(b).triangularView<Eigen::Upper>().
    solve( Z.rightCols(m-b) );
  t.stop(); double t3 = t.elapsed_time();

  
  t.start();
  T = W.transpose();
  t.stop(); t4 += t.elapsed_time();

  int n = A.cols();
  flops = 2.*m*n*b + 4.*m*b*b/3.0 + 1.*b*b*(m-b);


#ifdef PROF
  std::cout<<std::endl
    <<"--------------------\n"
    <<"  RandCPQR\n"
    <<"--------------------\n"
    <<"Rand:  "<<t0<<std::endl
    <<"GEMM:  "<<t1<<std::endl
    <<"CPQR:  "<<t2<<std::endl
    <<"Solve: "<<t3<<std::endl
    <<"Trans: "<<t4<<std::endl
    <<"--------------------\n"
    <<"Total: "<<t0+t1+t2+t3+t4<<std::endl
    <<"--------------------\n"
    <<std::endl;
#endif     
}

