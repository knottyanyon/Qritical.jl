# # Exercise 01

using DotEnv
ENVCFG = DotEnv.config(joinpath(ENV["PROJECT_ROOT"], ".env")); # Loads variables from .env 

EXDIR = joinpath(ENVCFG["EXERCISES_ROOT"], "01_SVD")

using LinearAlgebra
# ## 1. SVD a matrix

# Perform an SVD on the matrix $A$ given in Moodle as `A.txt` and find the Schmidt rank
# needed if singular values below $10^{−3}$ are discarded.

using DelimitedFiles

FPATH_A = joinpath(EXDIR, "A.txt")

A_mat = readdlm(FPATH_A)
tolerance = 10E-3
## Compute the singular value decomposition (SVD) of `A_mat` and return an SVD object.
A_svd = LinearAlgebra.svd(A_matrix); # Store the Factorization Object
typeof(A_svd)
Σ_cleaned = A_svd.S .> tolerance
rank_Schmidt = count(Σ_cleaned)

using Qritical
using CairoMakie

fig = draw_svd_hinton(A_mat, A_svd)

##
using ITensors

row_idx = Index(size(A_mat, 1), "row")
col_idx = Index(size(A_mat, 2), "col")

A_tensor = ITensor(A_matrix, row_idx, col_idx)
U, S, Vdag = ITensors.svd(A_tensor, row_idx)

@show ITensors.norm(U * S * Vdag - A_tensor)

# ## 2. SVD a state

# Perform an SVD on the state `psi.jls` given in Moodle. The format is a tensor of rank 10, dimensions $2^{10} = 1024$.
# Find the Schmidt rank needed if singular values below $10^{−6}$ are discarded for:

# ### (a) a bipartition of the system after the first site

# ### (b) a bipartition of the system in the middle

# ## 3. SVD an image

# Reproduce an SVD based image compression (use the image in Moodle or anything else
# you might like).

# *Hint:* Perform an SVD on each color channel.

## 4. Contractions

# Generate two random matrices $A, B$ each of size $N \times N$ and calculate the product
# $C_{i,j} = A_{i,k} B_{k,j}$,

# ### (a) once without using any libraries

# ### (b) once using a library of your choice

# # Sources
# [1] L. N. Trefethen, Numerical Linear Algebra (Society for Industrial and Applied Mathematics (SIAM, 3600 Market Street, Floor 6, Philadelphia, PA 19104), Philadelphia, Pa, 1997).
# [trefethen_1997](@cite)


