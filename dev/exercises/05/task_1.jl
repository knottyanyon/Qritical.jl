# # Task 5.1 — Vidal Notation

# !!! question "Task 5.1 — Vidal Notation"
#     Write a function that receives an MPS in left or right canonical form and
#     returns an MPS in ``\Gamma``-``\Lambda`` notation.

using LinearAlgebra, Qritical
#--

# ## The Vidal representation
#
# A left-canonical MPS stores site tensors ``A_i`` satisfying
# ``A_i^\dagger A_i = \mathbb{1}``.  The **Vidal** (Γ-Λ) form factors these as
#
# ```math
# A_i = \Lambda_{i-1}\, \Gamma_i
# ```
#
# where ``\Lambda_{i-1} = \operatorname{diag}(\lambda^{[i-1]}_1, \ldots)``
# is the diagonal matrix of singular values at bond ``i-1``.  Rearranging:
#
# ```math
# \Gamma_i = \Lambda_{i-1}^{-1} A_i, \qquad
# (\Gamma_i)_{\alpha\sigma\beta}
#     = \frac{(A_i)_{\alpha\sigma\beta}}{\lambda^{[i-1]}_\alpha}
# ```
#
# The boundary singular values are trivial: ``\Lambda_0 = \Lambda_L = [1]``.
# All the ``\Lambda`` arrays are already stored in `mps.bond_svs`:
# `bond_svs[i]` = ``\Lambda_{i-1}`` (singular values to the *left* of site ``i``).

function to_vidal(tensors::Vector{<:Array{<:Number,3}},
                  bond_svs::Vector{<:Vector{<:Real}})
    L = length(tensors)
    T = eltype(tensors[1])

    gammas  = Vector{Array{T, 3}}(undef, L)
    lambdas = [bond_svs[i + 1] for i in 1:L]   # lambdas[i] = Λ_i (right of site i)

    for i in 1:L
        A        = tensors[i]          # (χ_{i-1}, d, χ_i)
        Λ_prev   = bond_svs[i]         # Λ_{i-1}, length χ_{i-1}
        χL       = size(A, 1)
        gammas[i] = A ./ reshape(Λ_prev, χL, 1, 1)
    end

    return gammas, lambdas
end
#--

# ## Test on a left-canonical MPS

L = 10; χ = 8
mps = FiniteMPS(Spin{1//2}(), L, χ)
left_canonical_sweep!(mps)

tensors  = [t.data for t in mps.tensors]
bond_svs = mps.bond_svs

gammas, lambdas = to_vidal(tensors, bond_svs)
#--

# ## Verify round-trip: ``A_i = \Lambda_{i-1}\, \Gamma_i``
#
# Reconstructing left-canonical tensors from Γ and Λ must reproduce
# `tensors[i]` exactly (up to floating-point precision).

println("Round-trip errors ‖Λ_{i-1} Γ_i − A_i‖  (should be ≈ 0):")
for i in 1:L
    Λ_prev = bond_svs[i]
    χL     = length(Λ_prev)
    err    = norm(reshape(Λ_prev, χL, 1, 1) .* gammas[i] .- tensors[i])
    println("  Site $i: ", round(err; sigdigits=4))
end
#--

# ## Normalization check: ``\sum_\alpha \lambda^{[i]}_\alpha{}^2 = 1``
#
# For a normalized MPS the squared singular values at every bond sum to one.

println("\nBond norms ‖Λ_i‖² (should be ≈ 1 for a normalized state):")
for i in 1:L
    println("  Bond $i: ", round(sum(abs2, lambdas[i]); sigdigits=6))
end
#--
