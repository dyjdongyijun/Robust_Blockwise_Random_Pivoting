#include "randn.hpp"

#include "random.hpp"


void Generate_Gaussian(double *hA, int d, int n) {
  dvec Gmat(d*n);
  Random::Gaussian( Gmat, 0., 1. );

  //CHECK_CUDA( cudaMalloc((void **) &dA, d*n ) );
  thrust::copy_n( Gmat.begin(), d*n, hA );
}

