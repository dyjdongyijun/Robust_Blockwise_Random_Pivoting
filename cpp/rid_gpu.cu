#include "rid_gpu.hpp"
#include "util_gpu.hpp"


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


struct SchurIndex : public thrust::unary_function<int, int> {
  int a, m;

  __host__ __device__
    SchurIndex(int a_, int m_): a(a_), m(m_)  {}

  __host__ __device__
    int operator()(int i) {
      return i/a*m+i%a;
    }
};


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


  auto iter2 = thrust::make_transform_iterator(zero, SchurIndex(m, LD));
  auto elem2 = thrust::make_permutation_iterator(A, iter2);
  thrust::copy_n(B.begin(), m*n, elem2);
}


void rid_gpu(const double *hA, int m, int n, double tol, int blk) {

  // copy matrix to gpu
  dvec A(m*n);
  thrust::copy(hA, hA+m*n, A.begin());
  //print(A, m, n, "A");
 
  // allocate memory
  dvec LU(m*n); // same size as A
  double *ptrLU = thrust::raw_pointer_cast(LU.data());

  // (global) permutation
  ivec P(m);
  thrust::sequence(P.begin(), P.end(), 0);

  // random Gaussian matrix
  dvec G(n*blk);

  // Random_Gaussian_Matrix(G, 0, 1.0/blk);  
  thrust::counting_iterator<int> start(0);
  thrust::transform(start, start+G.size(), G.begin(), prg(0., 1.0/blk));
  //print(G, n, blk, "G");


  // compute sample matrix
  double one = 1.0, zero = 0.;
  cublasHandle_t blasHandle;
  CHECK_CUBLAS( cublasCreate(&blasHandle) )
  CHECK_CUBLAS( cublasDgemm(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, blk, n, &one,
        thrust::raw_pointer_cast(A.data()), m,
        thrust::raw_pointer_cast(G.data()), n, &zero,
        thrust::raw_pointer_cast(LU.data()), m) );
  //print(LU, m, n, "LU");
  

  int p  = std::min(m,n);
  int nb = std::ceil( p/blk );
  assert( p%blk == 0 );

  // prepare for LU factorizations
  cusolverDnHandle_t cusolverH = NULL;
  CUSOLVER_CHECK( cusolverDnCreate(&cusolverH) );

  dvec work; // working memory for LU
  ivec ipiv(blk); // local permutation
  ivec info(1); // an integer on gpu

  for (int i=0; i<nb; i++) {
    int k = i*blk; // number of processed rows/columns

    int a = m - k;
    int b = i < nb-1 ? blk : p-(nb-1)*blk;
    int lwork = 0;
    double *ptrE = ptrLU + k*m+k;
    CUSOLVER_CHECK( cusolverDnDgetrf_bufferSize(cusolverH, a, b, ptrE, m, &lwork) );

    work.resize(lwork);

    CUSOLVER_CHECK( cusolverDnDgetrf(cusolverH, a, b, ptrE, m, 
          thrust::raw_pointer_cast(work.data()), 
          thrust::raw_pointer_cast(ipiv.data()), 
          thrust::raw_pointer_cast(info.data()) ));

    //std::cout<<"a: "<<a<<", b: "<<b<<std::endl;
    //print(LU, m, n, "LU of new panel");
    //std::cout<<"info: "<<info[0]<<std::endl;
    assert( info[0]==0 );

    // global permuation (accumulation of local permuations)
    for (int j=0; j<b; j++) {
      int tmp = P[k+j];
      P[k+j] = P[ k+ipiv[j]-1 ];
      P[ k+ipiv[j]-1 ] = tmp;
    }

    // local permutation
    ivec Phat(a);
    thrust::sequence(Phat.begin(), Phat.end(), 0);
    for (int j=0; j<b; j++) {
      int tmp = Phat[j];
      Phat[j] = Phat[ ipiv[j]-1 ];
      Phat[ ipiv[j]-1 ] = tmp;
    }

    //print(ipiv, "ipiv");
    //print(Phat, "Phat");
    //print(P, "P");


    if (i>0) Permute_Matrix_Rows(Phat, ptrLU+k, a, k, m);
    //print(LU, m, n, "LU after local permutation");


    if (i == nb-1) break;

    // next panel
    b = i < nb-2 ? blk : p-(nb-1)*blk;
    k += blk; // number of processed rows/columns
    

    // randomized sketching
    thrust::counting_iterator<int> start(k*n);
    thrust::transform(start, start+G.size(), G.begin(), prg(0., 1.0/b));
    //print(G, n, blk, "G");
    
    double *ptrY = ptrLU + k*m;
    CHECK_CUBLAS( cublasDgemm(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
          m, b, n, &one,
          thrust::raw_pointer_cast(A.data()), m,
          thrust::raw_pointer_cast(G.data()), n, &zero,
          ptrY, m) );
  
    //print(LU, m, n, "new sample");

    // apply global permuation
    Permute_Matrix_Rows(P, ptrY, m, b, m);
    
    //print(LU, m, n, "Permute LU");

    // triangular solve
    double *ptrL = ptrLU;
    CHECK_CUBLAS( cublasDtrsm(blasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
          CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 
          k, b, &one,
          ptrL, m, 
          ptrY, m));

    //print(LU, m, n, "Triangular solve");

    // Schur complement
    double negone = -1.0;
    double *ptrS = ptrLU + k*m+k;
    CHECK_CUBLAS( cublasDgemm(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
          m-k, b, k, &negone,
          ptrL+k, m,
          ptrY, m, &one,
          ptrS, m) );

    //print(LU, m, n, "Schur complement");


    // compute Frobenius norm
    auto zero = thrust::make_counting_iterator<int>(0);
    auto S_idx = thrust::make_transform_iterator(zero, SchurIndex(m-k, m));
    auto S_elm = thrust::make_permutation_iterator(dptr(ptrS), S_idx);
    auto S_sqr = thrust::make_transform_iterator(S_elm, thrust::square<double>());
    double eSchur = thrust::reduce(S_sqr, S_sqr+(m-k)*b);

    eSchur = std::sqrt(eSchur);
    std::cout<<"Norm of Schur complement: "<<eSchur<<std::endl;
    if (eSchur < tol) break;

  }

}




