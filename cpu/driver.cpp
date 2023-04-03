#include <iostream>
#include <iomanip>      // std::setprecision

#include "matrix.hpp"
#include "rid.hpp"
#include "timer.hpp"


Mat FastDecay(int);
Mat Kahan(int);

int main(int argc, char *argv[]) {

#if 0
  int n = 1<<4;
  double tol = 1.e-6;
  int block = 2;
  
#else
  int n = 1<<10;
  double tol = 1.e-8;
  int block = 128;
#endif
  
 
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
      n = atoi(argv[++i]);
    if (!strcmp(argv[i], "-b"))
      block = atoi(argv[++i]);
    if (!strcmp(argv[i], "-tol"))
      tol = atof(argv[++i]);
  }
  std::cout
    <<"\nInputs:"
    <<"\n---------------------"
    <<"\nMatrix size:\t"<<n<<std::endl
    <<"block size:\t"<<block<<std::endl
    <<"tolerance:\t"<<tol<<std::endl
    <<"---------------------\n"<<std::endl;


  Eigen::HouseholderQR<Mat> qr1(Mat::Random(n,n)), qr2(Mat::Random(n,n));

  Mat U = qr1.householderQ();
  Mat V = qr2.householderQ();

  // fast decaying singular values
  Vec s(n);
  for (int i=0; i<n ;i++)
    s[i] = std::pow(1e-16, double(i)/(n-1));

  // test matrix
  Mat A = U*s.asDiagonal()*V.transpose();

  // initialization for randomized methods
  init_rand_generator();

  // new method
  Timer t; 
  std::vector<int> sk(n), rd(n);
  Mat T;
  double flops;
  double err;


#if 0
  int nb[7] = {8,    16,    32,    64,   128,   256,   512};
  for (int i=0; i<7; i++) {

    block = nb[i];
#endif

  // warm up
  RandAdapLUPP(A, sk, rd, T, flops, tol, block);
  
  t.start();
  RandAdapLUPP(A, sk, rd, T, flops, tol, block);
  t.stop();

  err = (A(rd,Eigen::all) - T*A(sk,Eigen::all)).norm();

  std::cout.precision(3);
  std::cout
    <<"\nResults:"
    <<"\n----------------------------------------------------------------------\n"
    <<"\t\ttime (s)\tGflop/s\t\trank\t\terror\n"
    <<"----------------------------------------------------------------------\n"
    <<"RandAdapLUPP\t"<<t.elapsed_time()
    <<"\t\t"<<flops/t.elapsed_time()/1.e9
    <<"\t\t"<<sk.size()
    <<"\t\t"<<err
    <<std::endl;

#if 0
  }
#endif

  // reference method (randomized LUPP with a given rank)
  int rank = sk.size();
  //for (; rank<n; rank++)
    //if (s[rank] < tol) break;
  
  sk.resize(rank), rd.resize(n-rank);
  T = Mat();
  flops = 0;
  err = 0.;
  
  // warm up
  RandLUPP(A, rank, sk, rd, T, flops);

  t.start();
  RandLUPP(A, rank, sk, rd, T, flops);
  t.stop();

  err = (A(rd,Eigen::all) - T*A(sk,Eigen::all)).norm();
  std::cout<<"RandLUPP\t"<<t.elapsed_time()
    <<"\t\t"<<flops/t.elapsed_time()/1.e9
    <<"\t\t"<<sk.size()
    <<"\t\t"<<err
    <<std::endl;


  // reference method (randomized CPQR with a given rank)
  sk.resize(rank), rd.resize(n-rank);
  T = Mat();
  flops = 0;
  err = 0.;
  
  //warm up
  RandCPQR(A, rank, sk, rd, T, flops);

  t.start();
  RandCPQR(A, rank, sk, rd, T, flops);
  t.stop();

  err = (A(rd,Eigen::all) - T*A(sk,Eigen::all)).norm();
  std::cout<<"RandCPQR\t"<<t.elapsed_time()
    <<"\t\t"<<flops/t.elapsed_time()/1.e9
    <<"\t\t"<<sk.size()
    <<"\t\t"<<err
    <<std::endl;


  Mat X = Mat::Random(n,n);
  Mat Y = Mat::Random(n,n);
  Mat Z = Mat::Zero(n,n);

  t.start();
  Z = X*Y;
  t.stop();
  std::cout<<"GEMM\t\t"<<t.elapsed_time()
    <<"\t\t"<<2.*n*n*n/t.elapsed_time()/1.e9
    <<std::endl;
  std::cout<<"----------------------------------------------------------------------\n\n";

  //std::cout<<"\nGPU:\n";
  //rid_gpu(A.data(), A.rows(), A.cols(), 1e-6, 2);

  return 0;
}


Mat FastDecay(int n) {

  Eigen::HouseholderQR<Mat> qr1(Mat::Random(n,n)), qr2(Mat::Random(n,n));

  Mat U = qr1.householderQ();
  Mat V = qr2.householderQ();

  // fast decaying singular values
  Vec s(n);
  for (int i=0; i<n ;i++)
    s[i] = std::pow(1e-16, double(i)/(n-1));

  // test matrix
  return U*s.asDiagonal()*V.transpose();
}

Mat Kahan(int n) {

  double z = 0.65;
  double p = std::sqrt(1-z*z);

  Mat U = -p * Mat::Ones(n, n);

  Vec d(n);
  for (int i=0; i<n; i++)
    d(i) = std::pow(z, i);

  Mat K = U.triangularView<Eigen::UnitUpper>();
  return d.asDiagonal() * K;
}


