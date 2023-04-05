#ifndef _rid_gpu_hpp_
#define _rid_gpu_hpp_


void RandAdapLUPP(const double *A, int m, int n,
    int *&sk, int *&rd, double *&T, int &, double &flops,
    double tol=1e-8, int blk=16);

void RandLUPP(const double *A, int m, int n, int rank,
    int *&, int *&, double *&, double&);


#include <vector>

void RandCPQR_column(const double *A, int m, int n, int rank,
    std::vector<int> &, std::vector<int> &, double *&, double&);


#endif
