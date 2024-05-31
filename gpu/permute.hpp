#ifndef _permute_hpp_
#define _permute_hpp_

#include "types.hpp"


void pivots_to_permutation(const ivec &, ivec &, int offset=0);

void Permute_Matrix_Rows(ivec &Perm, double *rawA, int m, int n, int LD);


#endif
