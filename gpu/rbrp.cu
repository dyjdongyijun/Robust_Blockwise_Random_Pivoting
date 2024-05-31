#include "rbrp.hpp"

#include "util.hpp"
#include "timer.hpp"
#include "types.hpp"
#include "flops.hpp"
#include "print.hpp"
#include "handle.hpp"
#include "submatrix.hpp"
#include "magma.h"


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


void RBRP(int m, int n, double *A, int blk,
    std::vector<int> &ranks, std::vector<double> &error, std::vector<double> &t_ranks,
    double &t_rbrp, double &flops) {

  std::cout<<"Enter RBRP\n";

  dvec X(m*n);
  thrust::copy_n(dptr(A), m*n, X.begin());
  double Anrm2 = thrust::transform_reduce( 
      X.begin(), X.end(), thrust::square<double>(), 0., thrust::plus<double>());

  //std::cout<<"finished initialization\n";
  //print(X, m, n, "X");

  // permutation vector
  ivec perm(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n));
  //print(perm, "initial permutation");

  // compute norms of columns
  dvec diags(n);
  for (int i=0; i<n; i++) {
    diags[i] = thrust::transform_reduce(
        X.begin()+i*m, X.begin()+(i+1)*m,
        thrust::square<double>(), 0., thrust::plus<double>());
  }
  //print(diags, "initial column norms (squared)");

  Timer t; t.start();

  int i = 0;
  int p = std::min(m, n);
  while (i < p) {
    int bs = std::min(blk, p-i);
    std::cout<<"i: "<<i<<", bs: "<<bs<<std::endl;

    // select a block of (global) indices
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
    //print(idx, "selected indices");


    // compute the (symmetric) difference between two index sets: 
    // (1) A = i:i+bs and (2) B = selected indices
    ivec Jin ( idx.size() ); // B \ A
    ivec Jout( idx.size() ); // A \ B
    {
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
    assert( Jin.size() == Jout.size() );
    //print(Jin,  "in indices");
    //print(Jout, "out indices");


    // swap columns of 'X'
    {
      // tmp = X(:, Jout)
      dvec tmp(m*Jout.size());
      auto zero  = thrust::make_counting_iterator<int>(0);
      auto iter1 = thrust::make_transform_iterator(zero, PermuteColumn(Jout.data(), m));
      auto elem1 = thrust::make_permutation_iterator( X.begin(), iter1 );
      thrust::copy_n( elem1, m*Jout.size(), tmp.begin() );
          
      // X(:, Jout) = X(:, Jin)
      auto iter2 = thrust::make_transform_iterator(zero, PermuteColumn(Jin.data(), m));
      auto elem2 = thrust::make_permutation_iterator( X.begin(), iter2 );
      thrust::copy_n( elem2, m*Jin.size(), elem1 );
    
      // X(:, Jin) = tmp
      thrust::copy_n( tmp.begin(), tmp.size(), elem2 );
    }
    //print(X, m, n, "X after swapping");


    // swap entries of 'diags'
    {
      auto zero = thrust::make_counting_iterator<int>(0);
      auto vOut = thrust::make_permutation_iterator( diags.begin(), Jout.begin() );
      auto vIn  = thrust::make_permutation_iterator( diags.begin(), Jin.begin()  );
      thrust::swap_ranges( vOut, vOut+Jout.size(), vIn );
      

      /*
      dvec tmp(bs);
      thrust::copy_n( diags.begin()+i, bs, tmp.begin() );
      //print(tmp, "tmp");
          
      auto iter = thrust::make_transform_iterator(zero, PermuteColumn(J.data(), 1));
      auto elem = thrust::make_permutation_iterator( diags.begin(), iter);
      thrust::copy_n(elem, bs, diags.begin()+i);
    
      thrust::copy_n(tmp.begin(), bs, elem);
      */
    }
    //print(diags, "diags after swapping");


    // swap entries of 'perm'
    {
      auto zero = thrust::make_counting_iterator<int>(0);
      auto vOut = thrust::make_permutation_iterator( perm.begin(), Jout.begin() );
      auto vIn  = thrust::make_permutation_iterator( perm.begin(), Jin.begin()  );
      thrust::swap_ranges( vOut, vOut+Jout.size(), vIn );
      

      /*
      ivec tmp(bs);
      thrust::copy_n( perm.begin()+i, bs, tmp.begin() );
          
      auto zero = thrust::make_counting_iterator<int>(0);
      auto iter = thrust::make_transform_iterator(zero, PermuteColumn(J.data(), 1));
      auto elem = thrust::make_permutation_iterator( perm.begin(), iter);
      thrust::copy_n(elem, bs, perm.begin()+i);
    
      thrust::copy_n(tmp.begin(), bs, elem);
      */
    }
    //print(perm, "perm after swapping");


    // computed pivots/permutation will be used later 
    int jpvt[bs] = {0};
    {
    // 1. CPQR on panel X(i:m,i:i+bs)
      int info;
      int nb = magma_get_dgeqp3_nb( m-i, bs );
      //int lwork = ( 3*bs+1 ) * nb;
      int lwork = 2*bs + ( bs+1 ) * nb;

      dvec tau(std::min(bs, m-i));
      dvec work(lwork);
      magmaDouble_ptr Xptr  = thrust::raw_pointer_cast( X.data()+i*m+i );
      magmaDouble_ptr dtau  = thrust::raw_pointer_cast( tau.data() );
      magmaDouble_ptr dwork = thrust::raw_pointer_cast( work.data() );
      magma_dgeqp3_gpu(m-i, bs, Xptr, m, jpvt, dtau, dwork, lwork, &info);
      assert( info==0 );
      //std::cout<<"lwork: "<<lwork<<std::endl;
        //<<"optimal work size: "<<work[0]<<std::endl;

      // 0-based indexing here
      for (int k=0; k<bs; k++) jpvt[k]--;
      /*
      std::cout<<"jpvt:\n";
      for (int k=0; k<bs; k++)
        std::cout<<jpvt[k]<<" ";
      std::cout<<std::endl;
      */

    // 2. apply Householder reflections to the trailing matrix
    //    which computes projection with new basis: R(I2,I3) = Q2' * A([I2 I3],I3)
                  
#if 1
    // query for workspace size      
      int lwork2 = -1;
      std::vector<double> htau( tau.size() );
      std::vector<double> hwork(1);
      thrust::copy_n(tau.begin(), htau.size(), htau.begin());
      magma_dormqr_gpu( 
          MagmaLeft, MagmaTrans,
          m-i, n-i-bs, bs, 
          Xptr, m, htau.data(), Xptr+bs*m, m, 
          hwork.data(), lwork2, dwork, nb, &info);

      lwork2 = int(hwork[0]);
      hwork.resize(lwork2);
      //std::cout<<"lwork2: "<<lwork2<<std::endl;
      
      magma_dormqr_gpu( 
          MagmaLeft, MagmaTrans,
          m-i, n-i-bs, bs, 
          Xptr, m, htau.data(), Xptr+bs*m, m, 
          hwork.data(), lwork2, dwork, nb, &info);
#else

      magma_dormqr_2stage_gpu(
          MagmaLeft, MagmaTrans,
          m-i, n-i-bs, bs,
          Xptr, m, Xptr+bs*m, m,
          dwork, nb, &info);
#endif
    }
    //print(X, m, n, "X after local HQR");


    // update (squared) norms of remaining columns
    {
      for (int k=i+bs; k<n; k++) {
        auto Xptr = X.begin() + k*m + i;
        diags[k] -= thrust::transform_reduce(
            Xptr, Xptr+bs,
            thrust::square<double>(), 0., thrust::plus<double>());
      }
    }
    //print(diags, "diags after updating");


    // permute R(1:i-1, i:i+bs)
    ivec djpvt(jpvt, jpvt+bs);
    {
      int  nrow = i;
      int  ncol = bs;
      dvec tmp(nrow*ncol);
      auto R = X.begin() + i*m;

      auto zero  = thrust::make_counting_iterator<int>(0);
      auto iter1 = thrust::make_transform_iterator(zero, SubMatrix(nrow, m));
      auto elem1 = thrust::make_permutation_iterator(R, iter1);
      thrust::copy_n(elem1, tmp.size(), tmp.begin());
      

      auto iter2 = thrust::make_transform_iterator(zero, PermuteColumn2(djpvt.data(), nrow, m));
      auto elem2 = thrust::make_permutation_iterator(R, iter2);
      thrust::copy_n(tmp.begin(), tmp.size(), elem2);
    }
    //print(X, m, n, "X after R permutation");
  
    
    // swap entries of 'perm'
    {
      ivec tmp(bs);
      thrust::copy_n( perm.begin()+i, bs, tmp.begin() );
          
      auto zero = thrust::make_counting_iterator<int>(0);
      auto iter = thrust::make_transform_iterator(zero, PermuteColumn(djpvt.data(), 1));
      auto elem = thrust::make_permutation_iterator( perm.begin()+i, iter);
      thrust::copy_n(tmp.begin(), bs, elem);
    }
    //print(perm, "perm after permutation");


    // next iteration
    i += bs;
    
  }

}


