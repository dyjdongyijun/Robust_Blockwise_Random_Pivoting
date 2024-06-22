
# Compilation instructions

A makefile is included, which requires
- Intel **icpc** compiler,
- the [**Eigen** library](https://eigen.tuxfamily.org/index.php?title=Main_Page) for matrix operations (on a CPU),
- the [**cuBLAS** library](https://docs.nvidia.com/cuda/cublas/) for matrix operations (on a GPU), and
- the [**MAGMA** library](https://icl.utk.edu/magma/) for computing QR decompositions with column pivoting (magma_dgeqp3_gpu). 

# Reproducibility

Results in Table 3 in [our manuscript](https://arxiv.org/pdf/2309.16002) can be reproduced by executing './driver -n 100000 -bs 64 -nb 10.'

