#include <iostream>

#include "matrix.hpp"
#include "rid.hpp"
#include "rid_gpu.hpp"


Mat FastDecay(int);
Mat Kahan(int);

int main(int argc, char *argv[]) {

  /*
  // problem size (assuming square matrices)
  int n = 64;
  Mat A = Kahan(n);
  //std::cout<<"A:\n"<<A<<std::endl;

  rid(A, 1e-8, 4);
  
  rid_gpu(A.data(), A.rows(), A.cols(), 1e-8, 4);
  */

  int n = 6;
  Mat A = FastDecay(n);
  std::cout<<"A:\n"<<A<<std::endl;
  
  rid(A, 1e-6, 2);
  
  rid_gpu(A.data(), A.rows(), A.cols(), 1e-6, 2);

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


