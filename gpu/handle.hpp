#ifndef _HANDLE_HPP_
#define _HANDLE_HPP_

#include "singleton.hpp"
#include "util.hpp"

#include <memory>

template<typename T>
class Singleton {
  friend class Handle_t; // access private constructor/destructor
  //friend class mgpuHandle_t;
public:
  static T& instance() {
    static const std::unique_ptr<T> instance{new T()};
    return *instance;
  }
private:
  Singleton() {};
  ~Singleton() {};
  Singleton(const Singleton&) = delete;
  void operator=(const Singleton&) = delete;
};


class Handle_t final: public Singleton<Handle_t>{
  friend class Singleton<Handle_t>; // access private constructor/destructor
private:
  Handle_t() {
    std::cout<<"Create Handle_t instance"<<std::endl;
    // sparse info
    //CHECK_CUSPARSE( cusparseCreateCsrgemm2Info(&info) )
    // sparse handle
    //CHECK_CUSPARSE( cusparseCreate(&sparse) )
    // matrix descriptor
    //CHECK_CUSPARSE( cusparseCreateMatDescr(&mat) )
    //CHECK_CUSPARSE( cusparseSetMatType(mat, CUSPARSE_MATRIX_TYPE_GENERAL) )
    //CHECK_CUSPARSE( cusparseSetMatIndexBase(mat, CUSPARSE_INDEX_BASE_ZERO) )
    // cublas handle
    CHECK_CUBLAS( cublasCreate(&blas) );
    CUSOLVER_CHECK( cusolverDnCreate(&solver) );
  }
public:
  ~Handle_t() {
    std::cout<<"Destroy Handle_t instance"<<std::endl;
    //CHECK_CUSPARSE( cusparseDestroyCsrgemm2Info(info) )
    //CHECK_CUSPARSE( cusparseDestroy(sparse) )
    //CHECK_CUSPARSE( cusparseDestroyMatDescr(mat) )
    CHECK_CUBLAS( cublasDestroy(blas) );
    //CUSOLVER_CHECK( cusolverDnDestroy(solver) );
    cusolverDnDestroy(solver);
  }
public:
  //csrgemm2Info_t info;
  //cusparseHandle_t sparse;
  //cusparseMatDescr_t mat;
  cublasHandle_t blas;
  cusolverDnHandle_t solver;
};

#endif
