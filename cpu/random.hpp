#ifndef _rand_matrix_hpp
#define _rand_matrix_hpp


class Random {
  public:
    static void Gaussian(double *, int n, double mean=0., double std=1.);
  private:
    static int skip;
};

#endif
