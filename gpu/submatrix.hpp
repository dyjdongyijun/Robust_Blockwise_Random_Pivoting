#ifndef _submatrix_hpp_
#define _submatrix_hpp_

#include "types.hpp"


struct SubMatrix : public thrust::unary_function<int, int> {
  int n, ld;

  __host__
    SubMatrix(int n_, int ld_): n(n_), ld(ld_)  {}

  __device__
    int operator()(int i) {
      return i/n*ld+i%n;
    }
};


#endif
