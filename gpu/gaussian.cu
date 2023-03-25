#include "gaussian.hpp"

#include <thrust/random.h>

struct prg : public thrust::unary_function<unsigned int, double> {
  double a, b;

  __host__ __device__
  prg(double _a=0.0, double _b=1.0) : a(_a), b(_b) {};

  __host__ __device__
  double operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::normal_distribution<double> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};


void Gaussian(dvec &v, double mean, double std) {
  thrust::counting_iterator<int> start(0);
  thrust::transform(start, start+v.size(), v.begin(), prg(mean, std));
}


