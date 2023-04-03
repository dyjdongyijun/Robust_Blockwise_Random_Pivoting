#include "random.hpp"

#include <thrust/random.h>
#include <thrust/host_vector.h>

#include <iostream>


struct prg : public thrust::unary_function<unsigned int, double> {
  double mean, std;

  __host__
  prg(double a=0.0, double b=1.0) {
    mean = a;
    std  = b;
  }

  __host__
  double operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::normal_distribution<double> dist(mean, std);
    rng.discard(n);
    return dist(rng);
  }
};


int Random::skip = 0;
void Random::Gaussian(double *A, int n, double mean, double std) {
  thrust::counting_iterator<int> start(skip);
  thrust::transform(start, start+n, A, prg(mean, std));
  skip += n;
}
