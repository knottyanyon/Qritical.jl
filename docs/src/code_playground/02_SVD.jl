using DotEnv
using DelimitedFiles
using LinearAlgebra
ENVCFG = DotEnv.config(joinpath(ENV["PROJECT_ROOT"], ".env")); # Loads variables from .env 

EXDIR = joinpath(ENVCFG["EXERCISES_ROOT"], "01_SVD")

FPATH_A = joinpath(EXDIR, "A.txt")

A_mat = readdlm(FPATH_A)

## Play around with the `svd` function and understand the arguments
## The argument `full` stands for "full SVD" or "reduced SVD".

## Calls LAPACK.gesdd
A_reduced_gesdd = LinearAlgebra.svd(A_mat; full=false, alg=LinearAlgebra.DivideAndConquer());
A_full_gesdd = LinearAlgebra.svd(A_mat; full=true, alg=LinearAlgebra.DivideAndConquer());

## Calls LAPACK.gesvd! (typically slower but more accurate)
A_reduced_gesvd = LinearAlgebra.svd(A_mat; full=false, alg=LinearAlgebra.QRIteration());
A_full_gesvd = LinearAlgebra.svd(A_mat; full=true, alg=LinearAlgebra.QRIteration());
