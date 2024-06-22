#ifndef _types_hpp_
#define _types_hpp_

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
//#include <thrust/reduce.h>
//#include <thrust/functional.h>
#include <thrust/binary_search.h>

template <typename T>
using tvec = thrust::device_vector<T>;

typedef tvec<int> ivec;
typedef tvec<double> dvec;


template <typename T>
using tptr = thrust::device_ptr<T>;

typedef tptr<int> iptr;
typedef tptr<double> dptr;

#endif
