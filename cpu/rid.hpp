#ifndef _rid_hpp_
#define _rid_hpp_

#include "matrix.hpp"

#include <vector>


void RandAdapLUPP(const Mat&, 
    std::vector<int>&, std::vector<int>&, Mat&, double&,
    double tol=1e-4, int blk=16);

void RandLUPP(const Mat&, int rank,
    std::vector<int>&, std::vector<int>&, Mat&, double&);

void RandCPQR(const Mat&, int rank,
    std::vector<int>&, std::vector<int>&, Mat&, double&);

Mat RandColSketch(const Mat&, int);

Mat RandRowSketch(const Mat&, int);

Mat RandMat(int, int, double, double);

void init_rand();


#endif
