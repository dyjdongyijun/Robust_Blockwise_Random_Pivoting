#include <iostream>
#include <vector>
#include <iomanip>      // std::setprecision

#include "cpqr.hpp"
#include "rbrp.hpp"

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

#if 1
  int n = 1<<3;
  int bs = 2;
  int nb = 2;
  
#else
  int n = 1<<10;
  int bs = 64;
  int nb = 6;
#endif
  
 
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
      n = atoi(argv[++i]);
    if (!strcmp(argv[i], "-bs"))
      bs = atoi(argv[++i]);
    if (!strcmp(argv[i], "-nb"))
      nb = atoi(argv[++i]);
  }
  std::cout
    <<"\nInputs:"
    <<"\n---------------------"
    <<"\nMatrix size:\t"<<n<<std::endl
    <<"block size:\t"<<bs<<std::endl
    <<"---------------------\n"<<std::endl;

  Timer t; 
  
  Mat A;
  {
    t.start();
    Vec s = Vec::Constant(n, 1.);
    int m = 100;
    for (int i=m+1; i<n ;i++) {
      if (i<m+51)
        s[i] = std::pow(0.8, i-m);
      else
        s[i] = 1e-5;
    }

    Eigen::HouseholderQR<Mat> qr1(Mat::Random(n,n)), qr2(Mat::Random(n,n));

    Mat U = qr1.householderQ();
    Mat V = qr2.householderQ();
    
    // test matrix
    A = U*s.asDiagonal()*V.transpose();
    t.stop();
  }
  std::cout<<"Create test matrix: "<<t.elapsed_time()<<" s\n";

  
  double *dA;
  Copy2Device(A.data(), n*n, dA);


  std::vector<int> sk(n), rd(n);
  int *d_sk=NULL, *d_rd=NULL;
  Mat T;
  double *dT=NULL;
  double flops;
  double err;

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

  /*
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
  int rank = r;
  //for (; rank<n; rank++)
    //if (s[rank] < tol) break;
  sk.resize(rank), rd.resize(n-rank);
  T = Mat::Zero(n-rank, rank);
  flops = 0;
  err = 0.;

  // warm-up
  RandLUPP(dA, n, n, rank, d_sk, d_rd, dT, flops);
  
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
  */


  //std::cout.precision(10);
  //std::cout<<"A:\n"<<A<<std::endl;
  //std::cout.precision(3);

  Copy2Device(A.data(), n*n, dA);

  
  std::vector<int> ranks(nb);
  for (int i=0; i<nb; i++)
    ranks[i] = bs * (i+1);

  std::vector<double> error(nb), t_ranks(nb);
  
  // CPQR on matrix A
  /*
  double t_cpqr;
  CPQR(n, n, dA, ranks, error, t_ranks, t_cpqr, flops);
  print(error, "error");
  print(t_ranks, "time");
  */

  // RBRP on matrix A
  /*
  double t_rbrp;
  RBRP(n, n, dA, bs, ranks, error, t_ranks, t_rbrp, flops);
  print(error, "error");
  print(t_ranks, "time");
  */

  // randLUPP
  {
    Copy2Device(A.data(), n*n, dA);
    for (int i=0; i<nb; i++) {
      int k = ranks[i];

      t.start();
      RandLUPP(dA, n, n, k, d_sk, d_rd, dT, flops);
      t.stop(); t_ranks[i] = t.elapsed_time();
     
      sk.resize(k);
      rd.resize(n-k);
      T = Mat::Zero(k, n-k);
      Copy2Host(d_sk, k, sk.data());
      Copy2Host(d_rd, n-k, rd.data());
      Copy2Host(dT, k*(n-k), T.data());
      error[i] = 
        (A(Eigen::all, rd) - A(Eigen::all, sk)*T).squaredNorm() /
        A.squaredNorm();    
    }
    std::cout<<"\n[RandLUPP]\n";
    print(error, "error");
    print(t_ranks, "time");
  }

  // randLUPP-OS
  {
    Copy2Device(A.data(), n*n, dA);
    for (int i=0; i<nb; i++) {
      int k = ranks[i];

      t.start();
      RandLUPP_OS(dA, n, n, k, d_sk, d_rd, dT, flops);
      t.stop(); t_ranks[i] = t.elapsed_time();
     
      sk.resize(k);
      rd.resize(n-k);
      T = Mat::Zero(k, n-k);
      Copy2Host(d_sk, k, sk.data());
      Copy2Host(d_rd, n-k, rd.data());
      Copy2Host(dT, k*(n-k), T.data());
      error[i] = 
        (A(Eigen::all, rd) - A(Eigen::all, sk)*T).squaredNorm() /
        A.squaredNorm();    

      //std::cout<<"T:\n"<<T<<std::endl;
    }
    std::cout<<"\n[RandLUPP-OS]\n";
    print(error, "error");
    print(t_ranks, "time");
  }

  std::cout<<"----------------------------------------------------------------------\n\n";


  return 0;
}


