#include "rid.hpp"
#include "timer.hpp"

#include <numeric> // std::iota
#include <iostream>                


void RandLUPP(const Mat &A, int rank, 
    std::vector<int>& sk, std::vector<int> &rd, Mat &T, double &flops) {

  Timer t;

  t.start();
  Mat R = RandMat(A.cols(), rank, 0., 1.);
  //Mat R = Mat::Zero(A.cols(), rank);
  //Random::Gaussian( R.data(), A.cols()*rank, 0., 1. );
  t.stop(); double t0 = t.elapsed_time();

  t.start();
  Mat Y = A*R;
  t.stop(); double t1 = t.elapsed_time();
  
  t.start();
  MKL_INT m = Y.rows();
  MKL_INT b = rank;
  MKL_INT info;
  MKL_INT ipiv[ b ];

  dgetrf( &m, &b, Y.data(), &m, ipiv, &info);
  assert( info==0 );
  t.stop(); double t2 = t.elapsed_time();

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

  t.start();
  T = Y.topRows(b).triangularView<Eigen::UnitLower>().
    solve<Eigen::OnTheRight>( Y.bottomRows(m-b) );
  t.stop(); double t3 = t.elapsed_time();

  int n = A.cols();
  flops = 2.*m*n*b + 2.*m*b*b/3.0 + 1.*b*b*(m-b);

#if 1
  std::cout<<std::endl
    <<"--------------------\n"
    <<"  RandLUPP\n"
    <<"--------------------\n"
    <<"Rand:  "<<t0<<std::endl
    <<"GEMM:  "<<t1<<std::endl
    <<"LUPP:  "<<t2<<std::endl
    <<"Solve: "<<t3<<std::endl
    <<"--------------------\n"
    <<"Total: "<<t0+t1+t2+t3<<std::endl
    <<"--------------------\n"
    <<std::endl;
#endif    
}

