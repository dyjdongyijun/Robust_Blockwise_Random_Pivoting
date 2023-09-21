# Robust Blockwise Random Pivoting
Code for our paper:
```
Robust Blockwise Random Pivoting: Fast and Accurate Adaptive Interpolative Decomposition
Yijun Dong, Chao Chen, Per-Gunnar Martinsson, Katherine Pearce. 
```

## Environment setup for Python
```
$ conda env create -f conda_env.yml -n rnla
$ conda activate rnla
$ pip install -r pip_pkg.txt
```

## File organization
```python
..
|-- main (this repo)
|   |-- python 
|   |   |-- RBRP.ipynb # main Python experiment demo
|   |   |-- gmm_dpp.npy # pre-computed results of k-DPP (and other methods) on a GMM matrix
|   |-- figs # output figures 
|   |-- conda_env.yml # conda environment
|   |-- pip_pkg.yml # pip packages
|-- dataset # data source (e.g., for torchvision)
```

## Reference
- [RPCholesky](https://github.com/eepperly/Randomly-Pivoted-Cholesky/tree/main)
- [DPPy](https://github.com/guilgautier/DPPy)
- [Help functions and data matrices](https://github.com/dyjdongyijun/Randomized_Subspace_Approximation)