#include "interp.hpp"
#include "util.hpp"
#include "handle.hpp"
#include "submatrix.hpp"


#if 1
struct PermuteColumn : public thrust::unary_function<int, int> {
  iptr P;
  int  m;

  __host__
    PermuteColumn(iptr P_, int m_):
      P(P_), m(m_)  {}

  __host__ __device__
    int operator()(int i) {
      int col = i/m; // after permutation
      int row = i%m;
      return row + P[col] * m;
    }
};

#else
struct PermuteColumn : public thrust::unary_function<int, int> {
  int *P;
  int  m;

  __host__
    PermuteColumn(int *P_, int m_):
      P(P_), m(m_)  {}

  __host__ __device__
    int operator()(int i) {
      int col = i/m; // after permutation
      int row = i%m;
      return row + P[col] * m;
    }
};
#endif


// This function is modified from the one in randLUPP_OS.cu.
// (So the interface is strange: be careful with the defintions of 'r' and 'k'.)
void Compute_interpolation_matrix(int r, int n, int k, const dvec &Y, ivec &P, double *T) {

  // permute columns of Yr
  dvec Ypmat(r*n); double *Yp = thrust::raw_pointer_cast(Ypmat.data());
  {
    auto zero = thrust::make_counting_iterator<int>(0);
    auto iter = thrust::make_transform_iterator(zero, PermuteColumn(P.data(), r));
    auto elem = thrust::make_permutation_iterator(Y.begin(), iter);
    thrust::copy_n(elem, r*n, dptr(Yp));
  }
  //print( P, "permutation" );
  //print( Ypmat, r, n, "Yp" );


  {
    dvec dtau(k); // k <= r
    double *tau = thrust::raw_pointer_cast(dtau.data());

    // query working space of geqrf and ormqr
    int lwork_geqrf = 0;
    auto const& handle = Handle_t::instance();
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(
          handle.solver, r, k, Yp, r, &lwork_geqrf));

    int lwork_ormqr = 0;
    CUSOLVER_CHECK(cusolverDnDormqr_bufferSize(
          handle.solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
          r, n-k, k,
          Yp, r, tau, Yp+r*k, r, &lwork_ormqr));

    int lwork = std::max(lwork_geqrf, lwork_ormqr);
    dvec dwork(lwork);
    ivec info(1);
    double *work = thrust::raw_pointer_cast(dwork.data());
    int   *dinfo = thrust::raw_pointer_cast(info.data());

    // QR factorization of Yp(:,1:k)
    CUSOLVER_CHECK(cusolverDnDgeqrf(
          handle.solver, r, k, Yp, r, tau, work, lwork, dinfo));
    assert( info[0]==0 );

    // apply Q'
    CUSOLVER_CHECK(cusolverDnDormqr(
          handle.solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
          r, n-k, k,
          Yp, r, tau, Yp+r*k, r, work, lwork, dinfo));
    assert( info[0]==0 );

    // solve with R
    double one = 1.;
    CUBLAS_CHECK(cublasDtrsm(
          handle.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
          CUBLAS_DIAG_NON_UNIT, k, n-k, &one, Yp, r, Yp+r*k, r));
  }

  auto ZERO = thrust::make_counting_iterator<int>(0);
  auto indx = thrust::make_transform_iterator(ZERO, SubMatrix(k, r));
  auto elem = thrust::make_permutation_iterator(dptr(Yp+r*k), indx);
  thrust::copy_n( elem, k*(n-k), dptr(T) );

}


