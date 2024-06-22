#include "rbrp.hpp"

#include "cpqr.hpp"
#include "util.hpp"
#include "timer.hpp"
#include "types.hpp"
#include "flops.hpp"
#include "print.hpp"
#include "handle.hpp"
#include "random.hpp"
#include "submatrix.hpp"
#include "magma.h"

#include <numeric>


// permute columns of a matrix
struct PermuteColumn : public thrust::unary_function<int, int> {
  iptr P;
  int  m;

  __host__
    PermuteColumn(iptr P_, int m_):
      P(P_), m(m_)  {}

  __device__
    int operator()(int i) {
      int col = i/m; // after permutation
      int row = i%m;
      return row + P[col] * m;
    }
};


// permute columns of a submatrix
struct PermuteColumn2 : public thrust::unary_function<int, int> {
  iptr P;
  int  m, ld;

  __host__
    PermuteColumn2(iptr P_, int m_, int ld_):
      P(P_), m(m_), ld(ld_)  {}

  __device__
    int operator()(int i) {
      int col = i/m; // after permutation
      int row = i%m;
      return row + P[col] * ld;
    }
};


struct Divide : public thrust::unary_function<int, int> {

  int n;

  __host__
    Divide(int n_) : n(n_) {}

  __device__
    int operator()(int i) {
      return i/n;
    }
};


/*
struct Thresher : public thrust::unary_function<double, int> {
  double _thresh;

  __host__ __device__ 
  Thresher(double thresh) : _thresh(thresh) {}

  __host__ __device__ 
    int operator()(double x) {
      return int(x > _thresh);
  }
};
*/


void column_norm_squared(dptr A, int m, int n, int ld, dvec &nrm2) {
  // A is a matrix of size m-by-n

  auto zero = thrust::make_counting_iterator<int>(0);
  auto idx  = thrust::make_transform_iterator(zero, Divide(m));

  //int  ncol = i+bs;
  //auto Xptr = X.begin() + ncol*m + i;
  auto itr  = thrust::make_transform_iterator(zero, SubMatrix(m, ld));
  auto val  = thrust::make_permutation_iterator( A, itr );
  auto sqr  = thrust::make_transform_iterator(val, thrust::square<double>());
  
  nrm2.resize(n);
  ivec col(n);
  auto end = thrust::reduce_by_key(
      idx, idx+m*n, sqr, col.begin(), nrm2.begin());
  assert( end.first-col.begin() == n );      
}


ivec select_indices(dvec cdf, int bs) {

#if 0
    ivec indices(thrust::make_counting_iterator(i), thrust::make_counting_iterator(n));
    {  
      // copy data for (in-place) sorting
      dvec tmp(n-i);
      thrust::copy_n( diags.begin()+i, n-i, tmp.begin() );
      thrust::sort_by_key( 
          tmp.begin(), tmp.end(), 
          indices.begin(), 
          thrust::greater<double>() ); // sort in descending order
    }
    ivec idx( indices.begin(), indices.begin()+bs );
#else
    
    // in-place cumulative sum
    thrust::inclusive_scan(cdf.begin(), cdf.end(), cdf.begin()); 
    //print(cdf, "scan");
    
    // normalize
    thrust::constant_iterator<double> s( cdf.back() );
    thrust::transform(cdf.begin(), cdf.end(), s, cdf.begin(), thrust::divides<double>());
    //print(cdf, "Sampling CDF");

    // uniform random numbers
    dvec unif(bs);
    Random::Uniform(unif, 0., 1.);
    //print(unif, "uniform numbers");
   
    ivec idx(bs);
    thrust::upper_bound(cdf.begin(), cdf.end(), unif.begin(), unif.end(), idx.begin());
    //print(idx, "Sampled index");

    thrust::sort(idx.begin(), idx.end());
    //print(idx, "Sorted index");

    auto ret = thrust::unique(idx.begin(), idx.end());
    idx.resize( ret-idx.begin() );
    //print(idx, "Unique index");

#endif

    return idx;
}
      

int robust_filter(const dptr dA, int m, int bs) {
  /*
     // attempt to do it on GPU
  {
    auto zero = thrust::make_counting_iterator<int>(0);
    auto row  = thrust::make_transform_iterator(zero, UpperTriangleRow(bs));
    auto idx  = thrust::make_transform_iterator(zero, UpperTriangleIdx(bs, m));
    auto val  = thrust::make_permutation_iterator( B.begin(), idx );
    auto sqr  = thrust::make_transform_iterator( val, thrust::square<double>() );

    ivec out_key(bs);
    dvec out_val(bs);
    auto res = thrust::reduce_by_key(
        row, row+bs*bs, sqr, out_key.begin(), out_val.begin());
    assert( res.first-out_key.begin()  == bs );
    assert( res.second-out_val.begin() == bs );
    
    print(out_key, "output key");
    print(out_val, "output value");

    dvec csum(bs);
    auto rit = thrust::make_reverse_iterator( out_val.begin() );
    thrust::inclusive_scan(rit, rit+bs, csum.begin());
    print(csum, "reverse cumulative sum");

    new_bs = thrust::transform_reduce(
        csum.begin(), csum.end(), Thresher(csum.back()/bs), 0, thrust::plus<int>());
  
  }
  */

  std::vector<double> R(bs*bs);
  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter = thrust::make_transform_iterator(zero, SubMatrix(bs, m));
  auto elem = thrust::make_permutation_iterator( dA, iter );
  thrust::copy_n(elem, bs*bs, R.data());

  //print(B, m, bs, "QR");
  //print(R, bs, bs, "R");

  std::vector<double> sqr(bs, 0.);
  for (int r=0; r<bs; r++) {
    for (int c=r; c<bs; c++) {
      int idx = r + c*bs;
      sqr[r] += R[idx] * R[idx];
    }
  }
  //print(sqr, "row square sum");
  
  std::vector<double> csum(bs, 0.);
  std::partial_sum(sqr.rbegin(), sqr.rend(), csum.begin());
  //print(csum, "cumulative sum");

  int k=0;
  for (; k<bs; k++)
    if (csum[k] >= csum.back()/bs)
      break;

  return bs-k;
}
    

void compute_symmetric_difference(int i, int bs, ivec idx, ivec &Jin, ivec &Jout) {
    // (1) A = i:i+bs-1 and (2) B = selected indices
    Jin.resize( idx.size() ); // B \ A
    Jout.resize( idx.size() ); // A \ B
    
    // sort the selected indices
    thrust::sort( idx.begin(), idx.end() );

    auto result =
      thrust::set_difference(
        thrust::make_counting_iterator<int>(i),
        thrust::make_counting_iterator<int>(i+bs),
        idx.begin(), idx.end(),
        Jout.begin() );
    Jout.resize( result-Jout.begin() );

    result =
      thrust::set_difference(
        idx.begin(), idx.end(),
        thrust::make_counting_iterator<int>(i),
        thrust::make_counting_iterator<int>(i+bs),
        Jin.begin() );
    Jin.resize( result-Jin.begin() );
}

   
void apply_permutation(int i, int bs, ivec idx, dvec &X, int m, ivec &perm, dvec &diags) {
    // compute the (symmetric) difference between two index sets: 
    // (1) A = i:i+bs-1 and (2) B = selected indices
    ivec Jin; // B \ A
    ivec Jout; // A \ B
    compute_symmetric_difference(i, bs, idx, Jin, Jout);
    //print(Jin,  "in indices");
    //print(Jout, "out indices");


    // swap columns of 'X' 
    {
      // selected columns
      dvec tmp(m*bs);
      auto zero  = thrust::make_counting_iterator<int>(0);
      auto iter1 = thrust::make_transform_iterator(zero, PermuteColumn(idx.data(), m));
      auto elem1 = thrust::make_permutation_iterator( X.begin(), iter1 );
      thrust::copy_n( elem1, m*bs, tmp.begin() );
          
      // X(:, Jin) = X(:, Jout)
      assert( Jin.size() == Jout.size() );
      auto iter2 = thrust::make_transform_iterator(zero, PermuteColumn(Jout.data(), m));
      auto elem2 = thrust::make_permutation_iterator( X.begin(), iter2 );
      
      auto iter3 = thrust::make_transform_iterator(zero, PermuteColumn(Jin.data(), m));
      auto elem3 = thrust::make_permutation_iterator( X.begin(), iter3 );
      thrust::copy_n( elem2, m*Jout.size(), elem3 );
    
      // X(:, i:i+bs-1) = tmp
      thrust::copy_n( tmp.begin(), tmp.size(), X.begin()+i*m );
    }
    //print(X, m, n, "X after swapping");

    
    // swap entries of 'perm'
    {
      ivec tmp(bs);
      auto val  = thrust::make_permutation_iterator( perm.begin(), idx.begin()  );
      thrust::copy_n( val, bs, tmp.begin() );
      
      assert( Jin.size() == Jout.size() );
      auto vIn  = thrust::make_permutation_iterator( perm.begin(), Jin.begin() );
      auto vOut = thrust::make_permutation_iterator( perm.begin(), Jout.begin() );
      thrust::copy_n( vOut, Jout.size(), vIn );

      thrust::copy_n( tmp.begin(), tmp.size(), perm.begin()+i );
    }
    //print(perm, "perm after swapping");


    // copy entries of 'diags'
    {
      auto vOut = thrust::make_permutation_iterator( diags.begin(), Jout.begin() );
      auto vIn  = thrust::make_permutation_iterator( diags.begin(), Jin.begin()  );
      thrust::copy_n( vOut, Jout.size(), vIn );
    }
    //print(diags, "diags after swapping");
}


void Householder_trail_matrix(int m, int n, int i, int bs, dptr B, dptr X, dptr tau) {

    int nrow = m-i;
    int ncol = bs;
    int nrhs = n-i-bs;
    int LD   = m;
    
    double *Xptr  = thrust::raw_pointer_cast( B + i );
    double *RHS   = thrust::raw_pointer_cast( X + i*m+i+bs*m );
      
    // for Householder reflections
    double *dtau  = thrust::raw_pointer_cast( tau );

    // query working space
    int lwork_ormqr = 0;
    auto const& handle = Handle_t::instance();
    CUSOLVER_CHECK( cusolverDnDormqr_bufferSize(
          handle.solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
          nrow, nrhs, ncol,
          Xptr, m, dtau, RHS, m, &lwork_ormqr));

    dvec work(lwork_ormqr);
    double *dwork = thrust::raw_pointer_cast( work.data() );


    ivec info(1);
    int *dinfo = thrust::raw_pointer_cast(info.data());
    CUSOLVER_CHECK( cusolverDnDormqr(
          handle.solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
          nrow, nrhs, ncol, 
          Xptr, LD, dtau, RHS, LD,
          dwork, lwork_ormqr, dinfo));
      assert( info[0]==0 );
}


void RBRP(int m, int n, double *A, int blk,
    std::vector<int> &ranks, std::vector<double> &error, std::vector<double> &t_ranks,
    double &t_rbrp, double &flops) {

  dvec X(m*n);
  thrust::copy_n(dptr(A), m*n, X.begin());
  //std::cout<<"finished initialization\n";
  //print(X, m, n, "X");

  
  Timer T; T.start();
  Timer t, ts;
  double t0 = 0., t1 = 0., t2 = 0., t3 = 0., t4 = 0.;

  // permutation vector
  ivec perm(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n));
  //print(perm, "initial permutation");

  // compute norms of columns
  t.start();
  dvec diags(n);
  column_norm_squared(X.data(), m, n, m, diags);
  double Anrm2 = thrust::reduce(diags.begin(), diags.end());
  t.stop(); t4 += t.elapsed_time();
  //print(diags, "initial column norms (squared)");


  int i = 0;
  int p = std::min(m, n);
  ranks.resize(0);
  error.resize(0);
  t_ranks.resize(0);
  auto const& handle = Handle_t::instance();
  
  while (i < p) {
    ts.start();
    int bs = std::min(blk, p-i);
    //std::cout<<"i: "<<i<<", bs: "<<bs<<std::endl;

    t.start();
    // select a block of (global) indices
    
    // sampling probability
    dvec prob(n-i);
    thrust::copy_n( diags.begin()+i, n-i, prob.begin() ); 
    //print(cdf, "probability");
   
    ivec idx = select_indices(prob, bs);
    //print(idx, "selected indices");
    
    thrust::constant_iterator<int> a( i );
    thrust::transform(idx.begin(), idx.end(), a, idx.begin(), thrust::plus<int>());
    //print(idx, "Sampled index");
    t.stop(); t3 += t.elapsed_time();


    
    t.start();
    bs = idx.size(); // in case not enough indices were sampled
    dvec B(m*bs);
    dvec tau( std::min(m-i, bs) );
    int jpvt[bs] = {0};
    {
      // CPQR on selected indices
      auto zero = thrust::make_counting_iterator<int>(0);
      auto iter = thrust::make_transform_iterator(zero, PermuteColumn(idx.data(), m));
      auto elem = thrust::make_permutation_iterator( X.begin(), iter );
      thrust::copy_n( elem, B.size(), B.begin() );
          

      int info;
      int lwork = 2*bs + ( bs+1 ) * magma_get_dgeqp3_nb( m-i, bs );

      dvec work(lwork);
      magmaDouble_ptr Bptr  = thrust::raw_pointer_cast( B.data() );
      magmaDouble_ptr dtau  = thrust::raw_pointer_cast( tau.data() );
      magmaDouble_ptr dwork = thrust::raw_pointer_cast( work.data() );
      magma_dgeqp3_gpu(m-i, bs, Bptr+i, m, jpvt, dtau, dwork, lwork, &info);
      assert( info==0 );

      // 0-based indexing here
      for (int k=0; k<bs; k++) jpvt[k]--;
      //print(jpvt, bs, "pivots of 1st CPQR");
    }
    t.stop(); t0 += t.elapsed_time();

    
    t.start();
    {
      // filter
      // adjust the block size
      bs = robust_filter(B.data()+i, m, bs);
      assert( bs>0 );
      //std::cout<<"new block size: "<<new_bs<<std::endl;
      
      // truncate selected indices
      ivec djpvt(jpvt, jpvt+bs);
      ivec copy = idx;
      auto itr  = thrust::make_permutation_iterator( copy.begin(), djpvt.begin() );
      thrust::copy_n(itr, bs, idx.begin());
      idx.resize(bs);
      //print(new_idx, "truncated indices");
    }
    t.stop(); t3 += t.elapsed_time();
    //print(idx, "selected indices");


    t.start();
    apply_permutation(i, bs, idx, X, m, perm, diags); 
    t.stop(); t1 += t.elapsed_time();
 

    // Apply Householder reflections to the trailing matrix
    //    which computes projection with new basis: R(I2,I3) = Q2' * A([I2 I3],I3)
    t.start();
    Householder_trail_matrix(m, n, i, bs, B.data(), X.data(), tau.data());
    t.stop(); t2 += t.elapsed_time();
    //print(X, m, n, "X after local HQR");


    // update (squared) norms of remaining columns
    t.start();
    {
      int  ncol = i+bs;
      dvec out(n-ncol);
      column_norm_squared(X.data()+ncol*m+i, bs, n-ncol, m, out);

      auto beg = diags.begin() + ncol;
      thrust::transform(beg, diags.end(), out.begin(), beg, thrust::minus<double>());
      
      //print(col, "column");
      //print(out, "result");
    }
    t.stop(); t4 += t.elapsed_time();
    error.push_back( thrust::reduce(diags.begin()+i+bs, diags.end()) / Anrm2 );
    //print(diags, "diags after updating");



    // next iteration
    i += bs;
 
    ranks.push_back( bs );

    ts.stop(); t_ranks.push_back( ts.elapsed_time() );
  }

  // cumulative sum
  for (int i=1; i<ranks.size(); i++) {
    ranks[i] += ranks[i-1];
    t_ranks[i] += t_ranks[i-1];
  }
  T.stop(); t_rbrp = T.elapsed_time();

//#ifdef PROF
#if 1
  std::cout<<std::endl
    <<"--------------------\n"
    <<"  RBRP\n"
    <<"--------------------\n"
    <<"CPQR:  "<<t0<<std::endl
    <<"Select indices:  "<<t3<<std::endl
    <<"Trail matrix:  "<<t2<<std::endl
    <<"Column norms:  "<<t4<<std::endl
    <<"permute:  "<<t1<<std::endl
    <<"--------------------\n"
    <<"Total: "<<t_rbrp<<std::endl
    <<"--------------------\n"
    <<std::endl;
#endif
}


