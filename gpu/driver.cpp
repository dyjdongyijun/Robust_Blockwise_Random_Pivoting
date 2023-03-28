#include <iostream>
#include <vector>
#include <iomanip>      // std::setprecision

#include "matrix.hpp"
#include "rid.hpp"
#include "gemm.hpp"
#include "timer.hpp"
#include "print.hpp"


// helper functions
void Copy2Device(double *hptr, int, double *&dptr);
void Copy2Host(double *dptr, int, double *hptr);
void Copy2Host(int *dptr, int, int *hptr);

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

  int rank = 0;
  for (; rank<n; rank++)
    if (s[rank] < tol) break;
  
  // test matrix
  Mat A = U*s.asDiagonal()*V.transpose();
  
  double *dA;
  Copy2Device(A.data(), n*n, dA);


  std::vector<int> sk(n), rd(n);
  int *d_sk=NULL, *d_rd=NULL;
  Mat T;
  double *dT=NULL;
  double flops;
  double err;

  Timer t; 
  
  std::cout.precision(3);
  std::cout
    <<"\nResults:"
    <<"\n----------------------------------------------------------------------\n"
    <<"\t\ttime (s)\tGflop/s\t\trank\t\terror\n"
    <<"----------------------------------------------------------------------\n";

  Mat X = Mat::Random(n,n);
  Mat Y = Mat::Random(n,n);
  Mat Z = Mat::Zero(n,n);

  double *dX, *dY, *dZ;
  Copy2Device(X.data(), n*n, dX);
  Copy2Device(Y.data(), n*n, dY);
  Copy2Device(Z.data(), n*n, dZ);
  
  // warm up
  GEMM(n, n, n, dX, dY, dZ);

  t.start();
  GEMM(n, n, n, dX, dY, dZ);
  t.stop();
  
  // check accuracy
  //Copy2Host(dZ, n*n, Z.data());
  //std::cout<<"Z:\n"<<Z<<std::endl;
  //std::cout<<"error: "<<(Z-X*Y).norm()<<std::endl;

  std::cout<<"GEMM\t\t"<<t.elapsed_time()
    <<"\t"<<2.*n*n*n/t.elapsed_time()/1.e9
    <<std::endl;

  // new method
  t.start();
  int r = 0; // computed rank
  RandAdapLUPP(dA, n, n, d_sk, d_rd, dT, r, flops, tol, block);
  t.stop();
 

  sk.resize(r);
  rd.resize(n-r);
  T = Mat::Zero(n-r, r);

  Copy2Host(d_sk, r, sk.data());
  Copy2Host(d_rd, n-r, rd.data());
  Copy2Host(dT, r*(n-r), T.data());
  err = (A(rd,Eigen::all) - T*A(sk,Eigen::all)).norm();
  std::cout<<"RandAdapLUPP\t"<<t.elapsed_time()
    <<"\t\t"<<flops/t.elapsed_time()/1.e9
    <<"\t\t"<<sk.size()
    <<"\t\t"<<err
    <<std::endl;


  // reference method (randomized LUPP with a given rank)
  sk.resize(rank), rd.resize(n-rank);
  T = Mat::Zero(n-rank, rank);
  flops = 0;
  err = 0.;

  t.start();
  RandLUPP(dA, n, n, rank, d_sk, d_rd, dT, flops);
  t.stop();

  Copy2Host(d_sk, rank, sk.data());
  Copy2Host(d_rd, n-rank, rd.data());
  Copy2Host(dT, rank*(n-rank), T.data());
  err = (A(rd,Eigen::all) - T*A(sk,Eigen::all)).norm();
  std::cout<<"RandLUPP\t"<<t.elapsed_time()
    <<"\t\t"<<flops/t.elapsed_time()/1.e9
    <<"\t\t"<<sk.size()
    <<"\t\t"<<err
    <<std::endl;

  //std::cout<<"T:\n"<<T<<std::endl;
  //std::cout<<"flops: "<<flops<<std::endl;


  // reference method (randomized CPQR with a given rank)
  // Note the interface is different from randLUPP
  Mat At = A.transpose();
  Copy2Device(At.data(), n*n, dA);
  sk.resize(rank), rd.resize(n-rank);
  T = Mat::Zero(rank, n-rank);
  flops = 0;
  err = 0.;

  // warm up magma
  RandCPQR_column(dA, n, n, rank, sk, rd, dT, flops);

  t.start();
  RandCPQR_column(dA, n, n, rank, sk, rd, dT, flops);
  t.stop();

  Copy2Host(dT, rank*(n-rank), T.data());
  err = (At(Eigen::all, rd) - At(Eigen::all, sk)*T).norm();
  std::cout<<"RandCPQR\t"<<t.elapsed_time()
    <<"\t\t"<<flops/t.elapsed_time()/1.e9
    <<"\t\t"<<sk.size()
    <<"\t\t"<<err
    <<std::endl;

  std::cout<<"----------------------------------------------------------------------\n\n";


  return 0;
}


