#include "rid.hpp"
#include "types.hpp"
#include "random.hpp"
#include "submatrix.hpp"
#include "handle.hpp"
#include "util.hpp"
#include "permute.hpp"


void RandAdapLUPP(const double *A, int m, int n, 
    int *&sk, int *&rd, double *&T, int &rank, double &flops,
    double tol, int blk) {

  //print(A, m, n, "A");
 
  // allocate memory
  dvec LUmat(m*n); // same size as A
  double *LU = thrust::raw_pointer_cast(LUmat.data());

  // (global) permutation
  ivec P(m);
  thrust::sequence(P.begin(), P.end(), 0);

  // random Gaussian matrix
  dvec Gmat(n*blk);
  Random::Gaussian(Gmat, 0., 1./blk);
  double *G = thrust::raw_pointer_cast(Gmat.data());
  //print(G, n, blk, "G");


  // compute sample matrix
  auto const& handle = Handle_t::instance();
  double one = 1.0, zero = 0.;
  CHECK_CUBLAS( cublasDgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
        m, blk, n, &one,
        A, m,
        G, n, &zero,
        LU, m) );
  flops = 2.*m*n*blk;
  //print(LU, m, n, "LU");
  

  int p  = std::min(m,n);
  int nb = std::ceil( p/blk );
  assert( p%blk == 0 );

  // prepare for LU factorizations
  dvec work; // working memory for LU
  ivec ipiv(blk); // local permutation
  ivec info(1); // an integer on gpu

  int k;
  for (int i=0; i<nb; i++) {
    k = i*blk; // number of processed rows/columns

    int a = m - k;
    int b = i < nb-1 ? blk : p-(nb-1)*blk;
    int lwork = 0;
    double *E = LU + k*m+k;
    CUSOLVER_CHECK( cusolverDnDgetrf_bufferSize(handle.solver, a, b, E, m, &lwork) );

    work.resize(lwork);
    CUSOLVER_CHECK( cusolverDnDgetrf(handle.solver, a, b, E, m, 
          thrust::raw_pointer_cast(work.data()), 
          thrust::raw_pointer_cast(ipiv.data()), 
          thrust::raw_pointer_cast(info.data()) ));

    assert( info[0]==0 );
    flops = flops + 2.*a*b*b/3.;
    //print(LU, m, n, "LU of new panel");


    // global permuation (accumulation of local permuations)
    pivots_to_permutation(ipiv, P, k);
    
    // local permutation
    ivec Phat(a);
    thrust::sequence(Phat.begin(), Phat.end(), 0);
    pivots_to_permutation(ipiv, Phat);

    //print(ipiv, "ipiv");
    //print(Phat, "Phat");
    //print(P, "P");


    if (i>0) Permute_Matrix_Rows(Phat, LU+k, a, k, m);
    //print(LU, m, n, "LU after local permutation");


    if (i == nb-1) break;

    // next panel
    b = i < nb-2 ? blk : p-(nb-1)*blk;
    k += blk; // number of processed rows/columns
    

    // randomized sketching
    Random::Gaussian(Gmat, 0., 1./b);
    //print(G, n, blk, "G");
    
    double *Y = LU + k*m;
    CHECK_CUBLAS( cublasDgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
          m, b, n, &one,
          A, m,
          G, n, &zero,
          Y, m) );
    flops = flops + 2.*m*n*b;
    //print(LU, m, n, "new sample");


    // apply global permuation
    Permute_Matrix_Rows(P, Y, m, b, m);
    
    //print(LU, m, n, "Permute LU");

    // triangular solve
    CHECK_CUBLAS( cublasDtrsm(handle.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
          CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 
          k, b, &one,
          LU, m, 
          Y, m));
    flops = flops + 1.*k*k*b;

    //print(LU, m, n, "Triangular solve");

    // Schur complement
    double negone = -1.0;
    double *S = LU + k*m+k;
    CHECK_CUBLAS( cublasDgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
          m-k, b, k, &negone,
          LU+k, m,
          Y, m, &one,
          S, m) );
    flops = flops + 2.*(m-k)*k*b;

    //print(LU, m, n, "Schur complement");


    // compute Frobenius norm
    auto zero = thrust::make_counting_iterator<int>(0);
    auto S_idx = thrust::make_transform_iterator(zero, SubMatrix(m-k, m));
    auto S_elm = thrust::make_permutation_iterator(dptr(S), S_idx);
    auto S_sqr = thrust::make_transform_iterator(S_elm, thrust::square<double>());
    double eSchur = thrust::reduce(S_sqr, S_sqr+(m-k)*b);

    eSchur = std::sqrt(eSchur);
    //std::cout<<"Norm of Schur complement: "<<eSchur<<std::endl;
    if (eSchur < tol) break;

  }


  CHECK_CUDA( cudaMalloc((void **) &sk, sizeof(int)*k) );
  CHECK_CUDA( cudaMalloc((void **) &rd, sizeof(int)*(m-k) ));
  CHECK_CUDA( cudaMalloc((void **) &T,  sizeof(double)*k*(m-k) ));

  thrust::copy_n( P.begin(), k, iptr(sk) );
  thrust::copy_n( P.begin()+k, m-k, iptr(rd) );


  CHECK_CUBLAS( cublasDtrsm(handle.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 
        m-k, k, &one,
        LU, m, 
        LU+k, m));

  
  auto Zero = thrust::make_counting_iterator<int>(0);
  auto indx = thrust::make_transform_iterator(Zero, SubMatrix(m-k, m));
  auto elem = thrust::make_permutation_iterator(dptr(LU+k), indx);
  thrust::copy_n( elem, k*(m-k), dptr(T) );  

  flops = flops + 1.*k*k*(m-k);
  
  rank = k;  // computed rank


#if 0
  std::cout<<std::endl
    <<"--------------------\n"
    <<"  RandAdapLUPP\n"
    <<"--------------------\n"
    <<"Alloc: "<<t4<<std::endl
    <<"Rand:  "<<t0<<std::endl
    <<"GEMM:  "<<t1<<std::endl
    <<"LUPP:  "<<t2<<std::endl
    <<"Solve: "<<t3<<std::endl
    <<"Copy:  "<<t5<<std::endl
    <<"Perm:  "<<t6<<std::endl
    <<"--------------------\n"
    <<"Total: "<<t0+t1+t2+t3+t4+t5+t6<<std::endl
    <<"--------------------\n"
    <<std::endl;
#endif  
}




