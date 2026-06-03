
# # Task 1.2 — SVD a matrix

# !!! question "Task 1.2"
#     Perform an SVD on the matrix ``A`` in `A.txt` and find the Schmidt rank
#     needed if singular values below ``10^{-3}`` are discarded.

DATA_ROOT = normpath(joinpath(@__FILE__, "..", "..", "data"))
FPATH_A   = joinpath(DATA_ROOT, "A.txt")
#--

using DelimitedFiles, LinearAlgebra, Qritical

A_mat = readdlm(FPATH_A)

# Attach a named upper index to each axis, then declare a row-vs-column
# bipartition.  `bipartition(left, A)` automatically forms the complement
# (the column index) so we only need to name the left side.

row = upper(:row, size(A_mat, 1))
col = upper(:col, size(A_mat, 2))
A   = IndexedTensor(A_mat, (row, col))
bp  = bipartition(Partition(row), A)

# [`KeepAbove(atol)`](@ref) retains every ``\sigma_i > \texttt{atol}``.
# The named-tuple return carries the factors plus the truncation error
# ``\varepsilon = \|\Sigma_\text{discarded}\|_2``.

res = tensor_svd(A, bp, KeepAbove(1e-3))

println("full rank:    ", minimum(size(A_mat)))
println("Schmidt rank: ", length(res.S))
println("ε ≈ ", round(res.ε; sigdigits=4))
#--

# The Eckart–Young–Mirsky theorem guarantees that the rank-``r`` truncated
# SVD ``A_r = U_r \Sigma_r V_r^\dagger`` is the *best* rank-``r`` matrix
# approximation of ``A`` in both the spectral and Frobenius norms:
#
# ```math
# \|A - A_r\|_F = \varepsilon = \left\| \begin{pmatrix}
#     \sigma_{r+1} \\ \vdots \\ \sigma_{\min(m,n)}
# \end{pmatrix} \right\|_2.
# ```
#
# We can verify this directly:

U, S, Vd = res.U, res.S, res.Vd
A_approx  = Matrix(U) * Diagonal(S) * Matrix(Vd)
frob_err  = norm(A_mat .- A_approx)
println("‖A - A_r‖_F ≈ ", round(frob_err; sigdigits=4), "  (should equal ε)")

# ## Notes

# ### Matrix norms

# !!! definition "Induced matrix norm"
#     Given vector norms ``\|\cdot\|_{(n)}`` on the domain and ``\|\cdot\|_{(m)}``
#     on the range of ``A \in \mathbb{C}^{m \times n}``, the induced matrix norm
#     ``\|A\|`` is the smallest ``C`` such that
#
#     ```math
#     \|Ax\|_{(m)} \leq C\|x\|_{(n)}, \quad \forall x \in \mathbb{C}^n.
#     ```
#     ``C`` is the maximum factor by which ``A`` can "stretch" a vector.

# !!! definition "Frobenius norm"
#     ```math
#     \|A\|_F = \left(\sum_{i,j} |a_{ij}|^2\right)^{1/2}
#             = \sqrt{\operatorname{tr}(A^* A)}.
#     ```
#     Equivalent to the Euclidean 2-norm of ``A`` viewed as an ``mn``-dimensional
#     vector. Both ``\|\cdot\|_2`` and ``\|\cdot\|_F`` are invariant under
#     multiplication by unitary matrices.

# ### Singular Value Decomposition

# !!! theorem "Norm identities"
#     ```math
#     \|A\|_2 = \sigma_1 \qquad \|A\|_F = \sqrt{\sigma_1^2 + \cdots + \sigma_r^2}.
#     ```

# !!! theorem "Eckart–Young–Mirsky (best low-rank approximation)"
#     The rank-``\nu`` truncated SVD ``A_\nu = \sum_{j=1}^{\nu} \sigma_j u_j v_j^*``
#     satisfies
#
#     ```math
#     \|A - A_\nu\|_2 = \sigma_{\nu+1}, \qquad
#     \|A - A_\nu\|_F = \sqrt{\sigma_{\nu+1}^2 + \cdots + \sigma_r^2}.
#     ```
#
#     No other rank-``\nu`` matrix achieves a smaller error in either norm.
