# # 1. SVD a matrix

# Perform an SVD on the matrix $A$ given in Moodle as `A.txt` and find the Schmidt rank
# needed if singular values below $10^{−3}$ are discarded.

DATA_ROOT = normpath(joinpath(@__FILE__, "..")) ## A workaround to ensure that the data can be read during local testing as well as pages deployment build

#--
FPATH_A = normpath(joinpath(DATA_ROOT, "A.txt"))
println("File path for A: $FPATH_A")

using Qritical: factorize_with_svd
using DelimitedFiles
A_mat = readdlm(FPATH_A);

tolerance = 10E-3;
# Compute the singular value decomposition (SVD) of `A_mat` 

#--
left_singular_mat, singular_mat, right_singular_mat = factorize_with_svd(A_mat; discard_below_threshold=true, threshold=tolerance);



# # Sources
# [1] L. N. Trefethen, Numerical Linear Algebra (Society for Industrial and Applied Mathematics (SIAM, 3600 Market Street, Floor 6, Philadelphia, PA 19104), Philadelphia, Pa, 1997).
# [trefethen_1997](@cite)