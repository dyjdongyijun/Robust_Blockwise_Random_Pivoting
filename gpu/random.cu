#include "random.hpp"

#include <thrust/random.h>
#include <chrono>


struct prg_g : public thrust::unary_function<unsigned int, double> {
  double mean, std;

  __host__ __device__
  prg_g(double a=0.0, double b=1.0) {
    mean = a;
    std  = b;
  }

  __host__ __device__
  double operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::normal_distribution<double> dist(mean, std);
    rng.discard(n);
    return dist(rng);
  }
};


struct prg_u : public thrust::unary_function<unsigned int, double> {
  double a, b;

  __host__ __device__
  prg_u(double a_=0., double b_=1.) :
  a(a_), b(b_) {}

  __host__ __device__
  double operator()(const unsigned int n) const {
    thrust::default_random_engine rng(1986);
    thrust::uniform_real_distribution<double> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};


int Random::skip = 0;
void Random::Gaussian(dvec &v, double mean, double std) {
  thrust::counting_iterator<int> start(skip);
  thrust::transform(start, start+v.size(), v.begin(), prg_g(mean, std));
  skip += v.size();
}

void Random::Uniform(dvec &v, double a, double b) {
  thrust::counting_iterator<int> start(skip);
  thrust::transform(start, start+v.size(), v.begin(), prg_u(a, b));
  skip += v.size();
}


