#include "util.hpp"

void CopyToGPU(double *hp, int n, double *&dp) {
  unsigned N = sizeof(double)*n;
  CHECK_CUDA( cudaMalloc((void **) &dp, N) );
  CHECK_CUDA( cudaMemcpy(dp, hp, N, cudaMemcpyHostToDevice) );
}

void CopyToCPU(double *dp, int n, double *hp) {
  thrust::device_ptr<double> dptr(dp);
  thrust::copy_n(dptr, n, hp);
}

