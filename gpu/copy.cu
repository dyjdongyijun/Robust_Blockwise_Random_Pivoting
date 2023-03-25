#include "util.hpp"
#include "types.hpp"

void Copy2Device(double *hp, int n, double *&dp) {
  unsigned N = sizeof(double)*n;
  CHECK_CUDA( cudaMalloc((void **) &dp, N) );
  CHECK_CUDA( cudaMemcpy(dp, hp, N, cudaMemcpyHostToDevice) );
}

void Copy2Host(double *dp, int n, double *hp) {
  thrust::device_ptr<double> dptr(dp);
  thrust::copy_n(dptr, n, hp);
}

void Copy2Host(int *dp, int n, int *hp) {
  thrust::device_ptr<int> dptr(dp);
  thrust::copy_n(dptr, n, hp);
}

