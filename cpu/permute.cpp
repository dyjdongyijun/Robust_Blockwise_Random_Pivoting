#include "permute.hpp"
#include "print.hpp"

#if 0
#include <thrust/host_vector.h>


struct PermuteRow : public thrust::unary_function<int, int> {
  const int *P;
  int a, m;

  __host__
    PermuteRow(const int *P_, int a_, int m_):
      P(P_), a(a_), m(m_)  {}

  __host__
    int operator()(int i) {
      return i/a*m + P[i%a];
    }
};


struct SubMatrix : public thrust::unary_function<int, int> {
  int n, ld;

  __host__
    SubMatrix(int n_, int ld_): n(n_), ld(ld_)  {}

  __host__
    int operator()(int i) {
      return i/n*ld+i%n;
    }
};



void Permute_Matrix_Rows(const int *P, double *A, int m, int n, int LD) {

  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter1 = thrust::make_transform_iterator(zero, PermuteRow(P, m, LD));
  auto elem1 = thrust::make_permutation_iterator(A, iter1);


  std::vector<double> B(m*n);
  thrust::copy_n(elem1, m*n, B.begin());

  //print(P, m, "P");
  //print(B, m, n, "B");


  auto iter2 = thrust::make_transform_iterator(zero, SubMatrix(m, LD));
  auto elem2 = thrust::make_permutation_iterator(A, iter2);
  thrust::copy_n(B.begin(), m*n, elem2);
}

#else


#include <vector>
#include <algorithm>


void Permute_Matrix_Rows(const int *P, double *A, int m, int n, int LD) {
  std::vector<double> B(m*n);

#pragma omp parallel for
  for (int j=0; j<n; j++) {
    for (int i=0; i<m; i++) {
      B[ P[i] + j*m ] = A[ i+j*LD ];
      //B[ i + j*m ] = A[ P[i]+j*LD ];
    }
  }

#pragma omp parallel for
  for (int j=0; j<n; j++) {
    for (int i=0; i<m; i++) {
      A[ i+j*LD ] = B[ i+j*m ];
    }
  }
}


#endif


