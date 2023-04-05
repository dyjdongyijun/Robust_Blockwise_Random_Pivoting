#include "rid.hpp"
#include "omp.h"

#include <EigenRand/EigenRand>
#include <iostream>
#include <random>



using PRNG = Eigen::Rand::Vmt19937_64;


int NT = omp_get_max_threads();
std::vector<PRNG> urng(NT);


void init_rand_generator() {
  std::random_device r;
  for (int i=0; i<NT; i++) {
    PRNG tmp{r()};
    urng[i] = tmp;
  }
}


Mat RandMat(int m, int n, double mean, double std) {

  Mat A(m, n);
  
#pragma omp parallel
  {  
    int nt = omp_get_num_threads();
    int i = omp_get_thread_num();
    int b = i < n%nt ? n/nt+1 : n/nt;
    int s = i < n%nt ? b*i : b*i+n%nt;
    //std::cout<<"thread "<<i<<", block: "<<b<<", start: "<<s<<std::endl;
    
    Mat B = Eigen::Rand::normal<Mat>(m, b, urng[i], mean, std);
    A.middleCols(s, b) = B;
  }
    
  return A;
}



