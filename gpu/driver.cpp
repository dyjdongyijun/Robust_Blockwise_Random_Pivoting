#include <iostream>
#include <vector>
#include <iomanip>      // std::setprecision
#include <random>                        

#include "id_cpqr.hpp"
#include "rbrp.hpp"
#include "randn.hpp"

#include "matrix.hpp"
#include "rid.hpp"
#include "gemm.hpp"
#include "timer.hpp"
#include "print.hpp"


// helper functions
void Copy2Device(double *hptr, int, double *&dptr);
void Copy2Host(double *dptr, int, double *hptr);
void Copy2Host(int *dptr, int, int *hptr);


Mat Matrix_GE(int n) {
  
#if 0
  int m = 0;
  double base = 0.2;
  double smin = 1e-5;
#else
  int m = 100;
  double base = 0.8;
  double smin = 1e-5;
#endif

  // Gaussian exponential decay
  Vec s = Vec::Constant(n, 1.);
  for (int i=m; i<n ;i++) {
      s[i] = std::max( std::pow(base, i-m), smin );
  }
  //std::cout<<"s:\n"<<s<<std::endl;

  Eigen::HouseholderQR<Mat> qr1(Mat::Random(n,n)), qr2(Mat::Random(n,n));

  Mat U = qr1.householderQ();
  Mat V = qr2.householderQ();
  
  // test matrix
  return U*s.asDiagonal()*V.transpose();
}

// sparsity parameter s = 0.1 by default
Mat Matrix_SNN(int n, double s=0.1) {
  Vec d(n);
  int m = 100;
  for (int i=0; i<n; i++) {
    if (i<m) d(i) = 10. / m;
    else d(i) = 1. / (i-m+1);
  }

  // generate two random sparse matrix
  int nnz = std::ceil(s * n); // number of nonzeros every column
  
  std::random_device rd;     
  std::mt19937 rng(rd());    
  std::uniform_int_distribution<int> uni(0, n-1); 
  std::uniform_real_distribution<double> unf(0., 1.); 

  typedef Eigen::Triplet<double> T;
  std::vector<T> listUD, listV;
  listUD.reserve(n*nnz);
  listV.reserve(n*nnz);

  for (int j=0; j<n; j++) {
    for (int k=0; k<nnz; k++) {
      int r = uni(rng), c = uni(rng);
      double u = unf(rng);
      listUD.push_back( T(r, c, u*d(c)) ); // multiply U with a diagonal matrix D
      
      r = uni(rng), c = uni(rng);
      double v = unf(rng);
      listV.push_back( T(r, c, v) );
    }
  }

  Eigen::SparseMatrix<double> UD(n, n), V(n,n);
  UD.setFromTriplets(listUD.begin(), listUD.end());
  V.setFromTriplets(listV.begin(), listV.end());

  //std::cout<<"UD:\n"<<UD<<std::endl;
  //std::cout<<"V:\n"<<V<<std::endl;
  return UD * V;
}
  
Mat Matrix_GMM(int n, int d) {
  // Gaussian mixed model
  //n = 100000;
  //int d = 1000;
  int k = 100;   // number of clusters
  assert( k <= d );

  int m = n / k; // points per cluster
  assert( d>0 && m>0 && "wrong parameters");
  
  Mat A = Mat::Zero(d, n);
  Generate_Gaussian(A.data(), d, n);
  for (int i=0; i<n; i++) {
    int j = i/m;
    A( j, i ) += 10*(j+1);
  }
  return A;
}


int main(int argc, char *argv[]) {

#if 0
  int n = 1<<4;
  int bs = 4;
  int nb = 2;
  
#else
  int n = 1000;
  int bs = 32;
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

  

  // create test matrix
  double *dA;
  Timer t; t.start();

  //Mat A = Matrix_GE(n);
  //Mat A = Matrix_SNN(n);
  Mat A = Matrix_GMM(n, 1000);

  //std::cout<<"A:\n"<<A<<std::endl;
   
  t.stop();
  std::cout<<"Create test matrix: "<<t.elapsed_time()<<" s\n";



  assert( n == A.cols() );
  int m = A.rows();
  std::cout.precision(3);
  std::cout
    <<"\nInputs:"
    <<"\n---------------------"
    <<"\nMatrix size:\t"<<m<<" x "<<n<<std::endl
    <<"block size:\t"<<bs<<std::endl
    <<"# blocks:\t"<<nb<<std::endl
    <<"---------------------\n"<<std::endl;
  // scientific format being consistent with Matlab default
  std::cout<<std::scientific;
  
  

  // RBRP on matrix A
  double t_rbrp;
  double flops;
  std::vector<int> ranks;
  std::vector<double> error, t_ranks;
 
  {
    Copy2Device(A.data(), m*n, dA);
    RBRP(m, n, dA, bs, ranks, error, t_ranks, t_rbrp, flops);
    std::cout<<"RBRP total time: "<<t_rbrp<<std::endl;

    // compare for first a few ranks
    ranks.resize(nb);
    error.resize(nb);
    t_ranks.resize(nb);
    print(ranks, "ranks");
    print(error, "error");
    print(t_ranks, "time");
  }
   

  /*
  ranks.resize(nb);
  for (int i=0; i<nb; i++)
    ranks[i] = (i+1)*bs;
  print(ranks, "ranks");
  */

  // CPQR on matrix A
  {
    std::cout<<"\n[Magma CPQR]\n";
    Copy2Device(A.data(), m*n, dA);
    double t_cpqr;
    CPQR(m, n, dA, ranks, error, t_ranks, t_cpqr, flops);
    std::cout<<"CPQR total time: "<<t_cpqr<<std::endl;
    print(error, "error");
    print(t_ranks, "time");
  }



  std::vector<int> sk(n), rd(n);
  int *d_sk=NULL, *d_rd=NULL;
  Mat T;
  double *dT=NULL;
  double err;

  /*
  // randCPQR
  {
    std::cout<<"\n[RandCPQR]\n";
    Copy2Device(A.data(), m*n, dA);
    for (int i=0; i<nb; i++) {
      int k = ranks[i];
      sk.resize(k);
      rd.resize(n-k);

      t.start();
      RandCPQR(dA, m, n, k, sk.data(), rd.data(), dT, flops);
      t.stop(); t_ranks[i] = t.elapsed_time();
     
      T = Mat::Zero(k, n-k);
      Copy2Host(dT, k*(n-k), T.data());
      error[i] = 
        (A(Eigen::all, rd) - A(Eigen::all, sk)*T).squaredNorm() /
        A.squaredNorm();    
    }
    print(error, "error");
    print(t_ranks, "time");
  }
  */

  // randCPQR-OS
  {
    std::cout<<"\n[RandCPQR-OS]\n";
    Copy2Device(A.data(), m*n, dA);
    for (int i=0; i<nb; i++) {
      int k = ranks[i];
      sk.resize(k);
      rd.resize(n-k);

      t.start();
      RandCPQR_OS(dA, m, n, k, sk.data(), rd.data(), dT, flops);
      t.stop(); t_ranks[i] = t.elapsed_time();
     
      T = Mat::Zero(k, n-k);
      Copy2Host(dT, k*(n-k), T.data());
      error[i] = 
        (A(Eigen::all, rd) - A(Eigen::all, sk)*T).squaredNorm() /
        A.squaredNorm();    
    }
    print(error, "error");
    print(t_ranks, "time");
  }

  /*
  // randLUPP
  {
    std::cout<<"\n[RandLUPP]\n";
    Copy2Device(A.data(), m*n, dA);
    for (int i=0; i<nb; i++) {
      int k = ranks[i];

      t.start();
      RandLUPP(dA, m, n, k, d_sk, d_rd, dT, flops);
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
    print(error, "error");
    print(t_ranks, "time");
  }
  */

  // randLUPP-OS
  {
    std::cout<<"\n[RandLUPP-OS]\n";
    Copy2Device(A.data(), m*n, dA);
    for (int i=0; i<nb; i++) {
      int k = ranks[i];

      t.start();
      RandLUPP_OS(dA, m, n, k, d_sk, d_rd, dT, flops);
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
    print(error, "error");
    print(t_ranks, "time");
  }



  return 0;
}


