# # Task 2.1 — Left Canonical State

# !!! question "Task 2.1 — Left Canonical State"
#     Write a function that performs a left canonical decomposition of the state
#     `psi.jls` of [Exercise 1](@ref "Task 1.3 — SVD a state").  The function
#     should allow for a maximum matrix dimension ``D`` to truncate the state
#     after each SVD.

using Serialization, LinearAlgebra, Qritical
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI = normpath(joinpath(DATA_ROOT, "psi.jls"))

ψ = deserialize(FPATH_PSI)   # shape (2, 2, …, 2): 10-site spin-1/2 state
L = ndims(ψ)
d = size(ψ, 1)               # local Hilbert space dimension = 2
println("State shape: ", size(ψ), "   L = $L   d = $d")
#--

# ## The left-canonical MPS decomposition
#
# An MPS in **left-canonical form** is a sequence of site tensors
# ``A_1, A_2, \ldots, A_L`` where each satisfies the *left-isometry* condition:
#
# ```math
# \sum_{\alpha_L, \sigma} (A_i)^*_{\alpha_L \sigma \alpha_R} \,
#                         (A_i)_{\alpha_L \sigma \alpha_R'} = \delta_{\alpha_R \alpha_R'}
# \qquad\Longleftrightarrow\qquad
# A_i^\dagger A_i = \mathbb{1}
# ```
#
# where the dagger ``{}^\dagger`` acts on the joint ``(\alpha_L, \sigma)`` index
# obtained by reshaping ``A_i`` to a matrix.
#
# The construction sweeps **left to right**.  At step ``i`` we reshape the
# current factor into a ``(\chi_{i-1} d) \times d^{L-i}`` matrix, take a thin
# SVD, keep at most ``D`` singular values, store the ``U`` factor as the
# left-canonical tensor ``A_i``, and absorb ``S V^\dagger`` into the remainder:
#
# ```math
# M_i \;=\; U_i \, S_i \, V_i^\dagger,\qquad
# A_i = \operatorname{reshape}(U_i,\, \chi_{i-1}, d, r_i),\qquad
# M_{i+1} = \operatorname{reshape}(S_i V_i^\dagger,\, r_i, d^{L-i})
# ```
#
# ``U_i^\dagger U_i = \mathbb{1}`` holds automatically from the SVD, so
# ``A_i`` is already left-canonical without any extra normalisation step.

function left_canonical_mps(ψ::Array, D::Int)
    L    = ndims(ψ)
    dims = size(ψ)           # (d, d, …, d)
    T    = eltype(ψ)
    RT   = real(T)

    tensors  = Vector{Array{T, 3}}(undef, L)
    bond_svs = Vector{Vector{RT}}(undef, L + 1)
    bond_svs[1]     = RT[1]
    bond_svs[L + 1] = RT[1]

    χL      = 1
    current = reshape(ψ, 1, :)   # (1, d^L)

    ## TODO: Implement the sequential left-to-right SVD loop.
    ##
    ## For i in 1:(L-1), do the following 5 steps:
    ##
    ##   1. Reshape `current` from (χL, d^(L-i+1)) to M (χL·dims[i], d^(L-i))
    ##   2. Thin SVD:  F = svd(M; full=false)
    ##   3. Truncate:  r = min(count(>(0), F.S), D)  — keep the r largest singular values
    ##   4. Store:     tensors[i] = reshape(F.U[:, 1:r], χL, dims[i], r)
    ##                 bond_svs[i+1] = F.S[1:r]
    ##   5. Advance:   current = Diagonal(F.S[1:r]) * F.Vt[1:r, :]    (shape r × d^(L-i))
    ##                 χL = r
    ##
    ## After the loop the last tensor absorbs whatever is left:
    ##   tensors[L] = reshape(current, χL, dims[L], 1)

    return tensors, bond_svs
end
#--

# ## Verifying left-isometry
#
# For a correctly constructed left-canonical MPS, every tensor reshaped as a
# matrix ``M_i = (\chi_{i-1} d) \times \chi_i`` must satisfy ``M_i^\dagger M_i = \mathbb{1}``.

function check_left_isometry(tensors::Vector{<:Array{<:Number,3}}; atol=1e-12)
    all_ok = true
    for (i, A) in enumerate(tensors)
        χL, d, χR = size(A)
        M   = reshape(A, χL * d, χR)
        err = norm(M' * M - I(χR))
        label = err < atol ? "✓" : "✗"
        println("  Site $i ($label):  ‖A†A − I‖ = $(round(err; sigdigits=4))")
        err > atol && (all_ok = false)
    end
    all_ok || @warn "Some sites are not left-isometric!"
    return all_ok
end
#--

# Decompose the 10-site state with maximum bond dimension ``D = 64``.
# Singular values below machine precision are treated as zero.

D = 64
tensors, bond_svs = left_canonical_mps(ψ, D)

println("Left-canonical MPS (D = $D):")
check_left_isometry(tensors)
#--

# ## Bond dimension profile
#
# The bond dimension ``\chi_i`` at bond ``i`` grows from 1 on the left boundary,
# reaches its maximum around the middle (where entanglement is highest), and
# shrinks back to 1 on the right boundary.  When ``D \ge`` the Schmidt rank at
# every bond, the decomposition is exact; otherwise information is discarded.

bond_dims = [size(t, 3) for t in tensors]
println("Bond dimensions: 1 → ", join(bond_dims, " → "))
println("Max bond dim used: ", maximum(bond_dims))
#--

# Now try smaller ``D`` values to see how truncation affects the state.
# A useful diagnostic is the truncation error at each bond: the discarded
# singular values measure how much spectral weight was lost.

for D_try in [1, 2, 4, 8]
    ts, svs = left_canonical_mps(ψ, D_try)
    errs = [sqrt(sum((s.^2)[D_try+1:end]) ) for s in svs[2:end-1] if length(s) > D_try]
    max_err = isempty(errs) ? 0.0 : maximum(errs)
    println("  D = $D_try:  bond dims = $(join([size(t,3) for t in ts], "·"))   ε_max ≈ $(round(max_err; sigdigits=3))")
end
#--

# ## Qritical.jl equivalent
#
# `left_canonical_sweep!` performs the same sweep on any `FiniteMPS`,
# whatever its current form.  The function updates `mps.bond_svs` at each
# bond and sets `mps.form = CanonicalForm(L, L+1)`.

mps = FiniteMPS(Spin{1//2}(), L, D)
left_canonical_sweep!(mps)

println("\nQritical.jl result:")
println("  form   : ", mps.form)
println("  ⟨ψ|ψ⟩  : ", round(real(overlap(mps, mps)); sigdigits=8))
println("  bond dims: ", join([size(t.data, 3) for t in mps.tensors], " · "))
#--
