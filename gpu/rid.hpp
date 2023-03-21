#ifndef _rid_gpu_hpp_
#define _rid_gpu_hpp_

#include <vector>


void rid_gpu(const double *A, int m, int n,
    std::vector<int> &sk, std::vector<int> &rd, double *&T, double &flops,
    double tol=1e-8, int blk=16);

void RandLUPP(const double *A, int m, int n, int rank,
    std::vector<int>&, std::vector<int>&, double *&, double&);

void RandCPQR(const double *A, int m, int n, int rank,
    std::vector<int>&, std::vector<int>&, double *&, double&);

void RandMat(int, int, double *&);

#endif
