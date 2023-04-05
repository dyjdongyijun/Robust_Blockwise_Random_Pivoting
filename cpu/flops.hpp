#ifndef _flops_hpp_
#define _flops_hpp_

// copy results from flops.h in the MAGMA package

// matmul
#define FMULS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))
#define FADDS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))

#define FLOPS_ZGEMM(m_, n_, k_) (6. * FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) + 2.0 * FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_CGEMM(m_, n_, k_) (6. * FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) + 2.0 * FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_DGEMM(m_, n_, k_) (     FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) +       FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_SGEMM(m_, n_, k_) (     FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) +       FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )


// LU
#define FMULS_GETRF(m_, n_) ( ((m_) < (n_)) \
    ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_) - 1. ) + (n_)) + (2. / 3.) * (m_)) \
    : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_) - 1. ) + (m_)) + (2. / 3.) * (n_)) )
#define FADDS_GETRF(m_, n_) ( ((m_) < (n_)) \
    ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_)      ) - (n_)) + (1. / 6.) * (m_)) \
    : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_)      ) - (m_)) + (1. / 6.) * (n_)) )

#define FLOPS_ZGETRF(m_, n_) (6. * FMULS_GETRF((double)(m_), (double)(n_)) + 2.0 * FADDS_GETRF((double)(m_), (double)(n_)) )
#define FLOPS_CGETRF(m_, n_) (6. * FMULS_GETRF((double)(m_), (double)(n_)) + 2.0 * FADDS_GETRF((double)(m_), (double)(n_)) )
#define FLOPS_DGETRF(m_, n_) (     FMULS_GETRF((double)(m_), (double)(n_)) +       FADDS_GETRF((double)(m_), (double)(n_)) )
#define FLOPS_SGETRF(m_, n_) (     FMULS_GETRF((double)(m_), (double)(n_)) +       FADDS_GETRF((double)(m_), (double)(n_)) )


// QR
#define FMULS_GEQRF(m_, n_) (((m_) > (n_)) \
    ? ((n_) * ((n_) * (  0.5-(1./3.) * (n_) + (m_)) +    (m_) + 23. / 6.)) \
    : ((m_) * ((m_) * ( -0.5-(1./3.) * (m_) + (n_)) + 2.*(n_) + 23. / 6.)) )
#define FADDS_GEQRF(m_, n_) (((m_) > (n_)) \
    ? ((n_) * ((n_) * (  0.5-(1./3.) * (n_) + (m_))           +  5. / 6.)) \
    : ((m_) * ((m_) * ( -0.5-(1./3.) * (m_) + (n_)) +    (n_) +  5. / 6.)) )

#define FLOPS_ZGEQRF(m_, n_) (6. * FMULS_GEQRF((double)(m_), (double)(n_)) + 2.0 * FADDS_GEQRF((double)(m_), (double)(n_)) )
#define FLOPS_CGEQRF(m_, n_) (6. * FMULS_GEQRF((double)(m_), (double)(n_)) + 2.0 * FADDS_GEQRF((double)(m_), (double)(n_)) )
#define FLOPS_DGEQRF(m_, n_) (     FMULS_GEQRF((double)(m_), (double)(n_)) +       FADDS_GEQRF((double)(m_), (double)(n_)) )
#define FLOPS_SGEQRF(m_, n_) (     FMULS_GEQRF((double)(m_), (double)(n_)) +       FADDS_GEQRF((double)(m_), (double)(n_)) )



#endif
