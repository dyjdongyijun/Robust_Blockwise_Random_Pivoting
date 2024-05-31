#ifndef rbrp_hpp
#define rbrp_hpp

#include <vector>

void RBRP(int m, int n, double *A, int blk,
    std::vector<int> &ranks, std::vector<double> &error, std::vector<double> &t_ranks,
    double &t_rbrp, double &flops);


#endif
