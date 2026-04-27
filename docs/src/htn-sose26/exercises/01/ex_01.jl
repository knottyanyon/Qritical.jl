# # Exercise 01



# using Qritical

# fig = draw_svd_hinton(A_mat, A_svd)

##
# using ITensors

# row_idx = Index(size(A_mat, 1), "row")
# col_idx = Index(size(A_mat, 2), "col")

# A_tensor = ITensor(A_matrix, row_idx, col_idx)
# U, S, Vdag = ITensors.svd(A_tensor, row_idx)

# @show ITensors.norm(U * S * Vdag - A_tensor)

# ## 2. SVD a state

# Perform an SVD on the state `psi.jls` given in Moodle. The format is a tensor of rank 10, dimensions $2^{10} = 1024$.
# Find the Schmidt rank needed if singular values below $10^{−6}$ are discarded for:

# ### (a) a bipartition of the system after the first site

# ### (b) a bipartition of the system in the middle

# ## 3. SVD an image

# Reproduce an SVD based image compression (use the image in Moodle or anything else
# you might like).

# *Hint:* Perform an SVD on each color channel.

# # Sources
# [1] L. N. Trefethen, Numerical Linear Algebra (Society for Industrial and Applied Mathematics (SIAM, 3600 Market Street, Floor 6, Philadelphia, PA 19104), Philadelphia, Pa, 1997).
# [trefethen_1997](@cite)
