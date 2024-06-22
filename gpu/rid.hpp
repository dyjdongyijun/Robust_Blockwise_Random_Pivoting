#ifndef _rid_gpu_hpp_
#define _rid_gpu_hpp_


void RandAdapLUPP(const double *A, int m, int n,
    int *&sk, int *&rd, double *&T, int &, double &flops,
    double tol=1e-8, int blk=16);

void RandLUPP(const double *A, int m, int n, int rank,
    int *&, int *&, double *&, double&);

void RandLUPP_OS(const double *A, int m, int n, int rank,
    int *&, int *&, double *&, double&);

// notice that h_sk and h_rd are pointers to host/cpu memmory
void RandCPQR(const double *A, int m, int n, int rank,
    int *h_sk, int *h_rd, double *&, double&);

void RandCPQR_OS(const double *A, int m, int n, int rank,
    int *h_sk, int *h_rd, double *&, double&);




#endif
