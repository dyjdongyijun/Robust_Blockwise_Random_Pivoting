#include "rid.hpp"
#include "types.hpp"
#include "gemm.hpp"
#include "gaussian.hpp"


void RandLUPP(const double *A, int m, int n, int k,
    int *&sk, int *&rd, double *&T, double &flops) {

  dvec Gmat(n*k);
  Gaussian(Gmat, 0., 1./std::sqrt(k));

  dvec Ymat(m*k);
  double *G = thrust::raw_pointer_cast( Gmat.data() );
  double *Y = thrust::raw_pointer_cast( Ymat.data() );
  GEMM(m, k, n, A, G, Y);

}

