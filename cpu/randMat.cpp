#include "rid.hpp"
#include "omp.h"

#include <EigenRand/EigenRand>
#include <iostream>

int NT = omp_get_max_threads();
std::vector<Eigen::Rand::Vmt19937_64> Gen(NT);


void init_rand() {
  for (int i=0; i<NT; i++) {
    Eigen::Rand::Vmt19937_64 gen{i};
    Gen[i] = gen;
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
    
    Mat B = Eigen::Rand::normal<Mat>(m, b, Gen[i], mean, std);
    A.middleCols(s, b) = B;
  }
    
  return A;
}


