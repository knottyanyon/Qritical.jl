
# # Task 1.2 — SVD a matrix

# !!! question "Task 1.2"
#     Perform an SVD on the matrix ``A`` in `A.txt` and find the Schmidt rank
#     needed if singular values below ``10^{-3}`` are discarded.

DATA_ROOT = normpath(joinpath(@__FILE__, "..", "..", "data"))
FPATH_A = joinpath(DATA_ROOT, "A.txt")
#--

using DelimitedFiles, LinearAlgebra, Qritical

A_mat = readdlm(FPATH_A)

# ## What did we just load?
#
# `readdlm` reads a whitespace-delimited text file and hands us back a plain
# Julia array. Before doing anything with it, it is worth peeking at the
# basics: how big is the matrix, and what type are the entries?

println("type:     ", typeof(A_mat))
println(
    "size:     ", size(A_mat), "  →  ", size(A_mat, 1), " rows × ", size(A_mat, 2), " cols"
)
println("elements: ", length(A_mat))
println("eltype:   ", eltype(A_mat))

# So we are looking at a square matrix of `Float64` entries (double precision —
# around 16 significant digits).  That is comfortable for SVD: LAPACK's `dgesdd`
# backward error is a small multiple of machine epsilon, so the smallest
# singular values we care about (above ``10^{-3}``) are perfectly reliable.
#--

# ## Thinking in tensor notation
#
# A matrix is a **valence-2 tensor** — one index for rows, one for columns.
# In the Penrose / tensor-network diagrammatic language, we draw it as a box
# with two legs sticking out.  The arrow direction on each leg tells us whether
# the index is *upper* (contravariant, incoming arrow) or *lower* (covariant,
# outgoing arrow):
#
# !!! note "Figure — A as a valence-2 tensor"
#     *Tensor diagram to be added here (excalidraw).*
#
# We will put both legs of ``A`` in the upper (contravariant) position for now.
# The distinction matters more once we start contracting MPS tensors and need
# to keep track of which legs can legally be contracted together.
#--

# ## Creating named, typed indices
#
# Plain Julia arrays have no leg labels — you have to remember externally that
# axis 1 is "rows" and axis 2 is "columns".  This is fine for a 2-leg tensor,
# but falls apart the moment you have an MPS site tensor with virtual bonds on
# both sides *and* a physical leg: three legs, easy to mix up.
#
# `Qritical.jl` gives each leg a `TIx` struct that packs three things together:
# a human-readable name, the index location (`Upper` or `Lower`), and the local imension.
# Two indices can only be contracted if one is upper and the other is lower
# *and* they share the same name. The type system enforces this at compile
# time, so positional mistakes surface as errors rather than silent wrong
# answers.
#
# Here we create one upper index for the row axis and one for the column axis:

row = upper(:row, size(A_mat, 1))
col = upper(:col, size(A_mat, 2))

println("row index: ", row)
println("col index: ", col)

# Wrapping the raw array in an `IndexedTensor` attaches these indices to the
# data without copying anything.  The `bipartition` call tells `tensor_svd`
# which legs go to the left factor (``U``) and which to the right (``V^\dagger``):

A = IndexedTensor(A_mat, (row, col))
bp = bipartition(Partition(row), A)

println("tensor indices: ", A.indices)
#--

# ## SVD in tensor notation
#
# What we are about to compute is a factorisation of ``A`` into three tensors
# connected by a new *virtual bond* (the Schmidt / singular-value index ``\chi``):
#
# ```math
# A = U \, \Sigma \, V^\dagger
# ```
#
# !!! note "Figure — SVD as a tensor diagram"
#     *SVD diagram to be added here (excalidraw). Show A splitting into U, Σ, Vd
#     with the bond index χ between them.*
#
# ## Truncation
#
# The singular values ``\sigma_1 \geq \sigma_2 \geq \cdots \geq 0`` decay from
# large (structure) to small (noise / redundancy).  Discarding those below a
# threshold compresses the representation.  [`KeepAbove(atol)`](@ref) keeps
# every ``\sigma_i > \texttt{atol}``; everything else is dropped.
#
# The *truncation error* ``\varepsilon = \|\Sigma_\text{discarded}\|_2`` is
# returned alongside the factors so we always know exactly how much we threw
# away.

res = tensor_svd(A, bp, KeepAbove(1e-3))

println("full rank:    ", minimum(size(A_mat)))
println("Schmidt rank: ", size(res.Σ.data, 1))
println("ε ≈ ", round(res.ε; sigdigits=4))
#--

# ## The result — IndexedTensors with named bonds
#
# `tensor_svd` does not return bare matrices.  `U`, `Σ`, and `Vd` come back as
# `IndexedTensor`s: the original partition legs are re-attached, and fresh
# `TIx{Lower}` / `TIx{Upper}` bond legs with derived labels are added at the
# shared interfaces:

println("typeof(res.U):  ", typeof(res.U))
println("U  indices: ", res.U.indices)
println("Vd indices: ", res.Vd.indices)

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
# We can verify this by reconstructing ``A_r`` from the `.data` fields (using
# `.data` keeps the multiplication in raw array space, avoiding any ambiguity
# about how `IndexedTensor * Diagonal` dispatches):

U, Σ, Vd = res.U, res.Σ, res.Vd
A_approx = U.data * Σ.data * Vd.data
frob_err = norm(A_mat .- A_approx)
println("‖A - A_r‖_F ≈ ", round(frob_err; sigdigits=4), "  (should match ε above)")

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
