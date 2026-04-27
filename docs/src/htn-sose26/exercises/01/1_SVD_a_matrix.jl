# # 1. SVD a matrix

# Perform an SVD on the matrix $A$ given in Moodle as `A.txt` and find the Schmidt rank
# needed if singular values below $10^{−3}$ are discarded.

# Loads variables from .env 
using DotEnv
ENVCFG = DotEnv.config(joinpath(ENV["PROJECT_ROOT"], ".env"));

EXDIR = joinpath(ENVCFG["EXERCISES_ROOT"], "01")
FPATH_A = joinpath(EXDIR, "A.txt")

println("File path for A: $FPATH_A")

using LinearAlgebra
using DelimitedFiles

A_mat = readdlm(FPATH_A)

tolerance = 10E-3

# Compute the singular value decomposition (SVD) of `A_mat` and return an SVD object.
A_svd = LinearAlgebra.svd(A_mat);

# Store the Factorization Object

typeof(A_svd)
Σ_cleaned = A_svd.S .> tolerance
rank_Schmidt = count(Σ_cleaned)
