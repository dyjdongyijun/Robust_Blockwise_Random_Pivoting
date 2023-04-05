#include "rid.hpp"

#include <EigenRand/EigenRand>

Eigen::Rand::Vmt19937_64 generator;

Mat RandColSketch(const Mat &A, int b) {

  // normal distribution with mean = 0, stdev = 1/b
  Mat R = Eigen::Rand::normal<Mat>(A.cols(), b, generator, 0.0, 1.0/b);

  return A*R;
}

Mat RandRowSketch(const Mat &A, int b) {

  // normal distribution with mean = 0, stdev = 1/b
  Mat R = Eigen::Rand::normal<Mat>(b, A.rows(), generator, 0.0, 1.0/b);

  return R*A;
}

