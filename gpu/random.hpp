#ifndef _rand_matrix_hpp
#define _rand_matrix_hpp

#include "util.hpp"
#include "types.hpp"

class Random {
  public:
    static void Gaussian(dvec&, double mean=0., double std=1.);
    static void Uniform(dvec&, double a=0., double b=1.);
  private:
    static int skip;
};

#endif
