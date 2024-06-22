#ifndef cpqr_hpp
#define cpqr_hpp

#include <vector>

void CPQR(int m, int n, double *A, 
    const std::vector<int> ranks, std::vector<double> &err, std::vector<double> &t_ranks, 
    double &t_cpqr, double &flops);

#endif
