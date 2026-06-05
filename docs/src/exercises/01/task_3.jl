
# # Task 1.3 — SVD a state

# !!! question "Task 1.3"
#     Perform an SVD on the state `psi.jls` — a rank-10 tensor with shape
#     ``(2, 2, \ldots, 2)`` representing a 10-qubit state. Find the Schmidt rank
#     needed if singular values below ``10^{-6}`` are discarded for:
#     - **(a)** a bipartition after the first site;
#     - **(b)** a bipartition at the middle (5 | 5).

DATA_ROOT = normpath(joinpath(@__FILE__, "..", "..", "data"))
FPATH_PSI = joinpath(DATA_ROOT, "psi.jls")
#--

using Serialization, LinearAlgebra, Qritical, CairoMakie

ψ = deserialize(FPATH_PSI)    # shape (2,2,...,2): 10 sites, each dim-2
N = ndims(ψ)

# Attach a named upper index to each site leg.  All sites share dimension 2.
# Using `Symbol(:s, i)` gives labels `:s1, :s2, …, :s10`.

sites = [upper(Symbol(:s, i), 2) for i in 1:N]
A     = IndexedTensor(ψ, Tuple(sites))
#--

# ## Part (a) — bipartition after the first site (1 | 9)

bp_a  = bipartition(Partition(sites[1]), A)
res_a = tensor_svd(A, bp_a, KeepAbove(1e-6); normalize=true)
(bipartition = "1 | 9", Schmidt_rank = size(res_a.Σ.data, 1))
#--

# ## Part (b) — bipartition at the middle (5 | 5)

bp_b  = bipartition(Partition(sites[1:N÷2]...), A)
res_b = tensor_svd(A, bp_b, KeepAbove(1e-6); normalize=true)
(bipartition = "5 | 5", Schmidt_rank = size(res_b.Σ.data, 1))
#--

# ## Entanglement entropy profile

# The **von Neumann entanglement entropy** of a bipartition quantifies how much
# the two halves are correlated.  Given Schmidt coefficients
# ``\lambda_i`` (normalised singular values with ``\sum_i \lambda_i^2 = 1``),
# the entropy is:
#
# ```math
# S = -\sum_i \lambda_i^2 \log_b \lambda_i^2,
# ```
#
# with base ``b = 2`` for **bits** or ``b = e`` for **nats**.
# A product state has ``S = 0``; a maximally entangled state across a
# 2^k|2^k bipartition has ``S = k`` bits.
#
# **Your implementation:** fill in the formula below.
# - Choose a logarithm base (and note what units the result is in).
# - Guard against ``\lambda_i = 0`` to avoid ``0 \log 0``.

function entanglement_entropy(λ::AbstractVector{<:Real})
    p = λ .^ 2
    return -sum(pᵢ * log2(pᵢ) for pᵢ in p if pᵢ > 0)
end
#--

# Once `entanglement_entropy` is implemented, the cell below sweeps every
# bipartition boundary and plots the resulting entropy profile.

entropies = map(1:N-1) do i
    bp  = bipartition(Partition(sites[1:i]...), A)
    res = tensor_svd(A, bp, KeepMachineEps(); normalize=true)
    entanglement_entropy(diag(res.Σ.data))
end

fig = Figure(size=(620, 360))
ax  = Axis(fig[1, 1];
    title  = "Entanglement entropy across bipartitions",
    xlabel = "boundary position  i  (site i | i+1 … N)",
    ylabel = "S (bits)",
    xticks = 1:N-1,
)
lines!(ax, 1:N-1, entropies; color=:teal, linewidth=2.5)
scatter!(ax, 1:N-1, entropies; color=:teal, markersize=9)
fig

# ## Notes

# - A small Schmidt rank means the bipartition carries *low entanglement*: the
#   state can be represented faithfully with few terms in its Schmidt expansion.
# - The entropy at the middle bipartition is typically the highest because the
#   two halves can exchange information across every pair of consecutive bonds
#   between them.
# - Both `normalize=true` and `KeepMachineEps()` matter here: the former
#   produces proper Schmidt coefficients ``\sum_i \lambda_i^2 = 1``, while the
#   latter discards floating-point noise that would otherwise inflate the
#   entropy by tiny non-zero terms.
