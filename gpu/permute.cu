#include "permute.hpp"
#include "submatrix.hpp"

#include <vector>
#include <numeric>      // std::iota


void pivots_to_permutation(const ivec &Dpiv, ivec &P, int ofs) {
  unsigned k = Dpiv.size();
  std::vector<int> Hpiv(k);
  thrust::copy_n( Dpiv.begin(), k, Hpiv.begin() );
 
  unsigned m = P.size();
  std::vector<int> HP(m);
  thrust::copy_n( P.begin(), m, HP.begin() );
  for (int j=0; j<k; j++) {
    int tmp = HP[ j+ofs ];
    HP[ j+ofs ] = HP[ Hpiv[j]-1+ofs ];
    HP[ Hpiv[j]-1+ofs ] = tmp;
  }

  thrust::copy_n( HP.begin(), m, P.begin() );
}


struct PermuteRow : public thrust::unary_function<int, int> {
  iptr P;
  int a, m;

  __host__
    PermuteRow(iptr P_, int a_, int m_):
      P(P_), a(a_), m(m_)  {}

  __device__
    int operator()(int i) {
      return i/a*m + P[i%a];
    }
};


void Permute_Matrix_Rows(ivec &Perm, double *rawA, int m, int n, int LD) {

  iptr P = Perm.data();
  dptr A(rawA);

  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter1 = thrust::make_transform_iterator(zero, PermuteRow(P, m, LD));
  auto elem1 = thrust::make_permutation_iterator(A, iter1);


  dvec B(m*n);
  thrust::copy_n(elem1, m*n, B.begin());
  //print(B, m, n, "B");


  auto iter2 = thrust::make_transform_iterator(zero, SubMatrix(m, LD));
  auto elem2 = thrust::make_permutation_iterator(A, iter2);
  thrust::copy_n(B.begin(), m*n, elem2);
}

