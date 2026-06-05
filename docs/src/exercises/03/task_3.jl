# # Task 3.3 — Checking the Normalization

# !!! question "Task 3.3 — Checking the Normalization"
#     Write a function that receives an MPS and checks at each site for left
#     (``A``) and right (``B``) normalization.  Think about a good measure
#     telling you how far away you are from unity in these normalizations.

using Serialization, LinearAlgebra, Qritical
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))

ψ = deserialize(normpath(joinpath(DATA_ROOT, "psi.jls")))
L = ndims(ψ)
d = size(ψ, 1)
#--

# ## What does "distance from isometry" mean?
#
# A tensor ``T`` of shape ``(\chi_L, d, \chi_R)`` is:
#
# - **Left-canonical** if ``M^\dagger M = \mathbb{1}_{\chi_R}`` where
#   ``M = \operatorname{reshape}(T, \chi_L d, \chi_R)``
# - **Right-canonical** if ``M M^\dagger = \mathbb{1}_{\chi_L}`` where
#   ``M = \operatorname{reshape}(T, \chi_L, d \chi_R)``
#
# A natural scalar measure is the Frobenius norm of the deviation:
# ``\delta_L(T) = \|M^\dagger M - \mathbb{1}\|_F`` and
# ``\delta_R(T) = \|M M^\dagger - \mathbb{1}\|_F``.
# Both equal zero for a perfectly isometric tensor and grow with the
# deviation.

function check_normalization(tensors::Vector{<:Array{<:Number,3}})
    N = length(tensors)
    left_errors  = Vector{Float64}(undef, N)
    right_errors = Vector{Float64}(undef, N)
    println("  site │  δ_L (left iso)   │  δ_R (right iso)")
    println("  ─────┼───────────────────┼──────────────────")
    for (i, T) in enumerate(tensors)
        χL, d, χR = size(T)
        ML = reshape(T, χL * d, χR)
        MR = reshape(T, χL, d * χR)
        left_errors[i]  = norm(ML' * ML - I(χR))
        right_errors[i] = norm(MR * MR' - I(χL))
        println("  $(lpad(i,4)) │  $(rpad(round(left_errors[i]; sigdigits=3), 17)) │  $(round(right_errors[i]; sigdigits=3))")
    end
    return (left_errors=left_errors, right_errors=right_errors)
end
#--

# ## Test on known canonical forms
#
# A freshly left-canonicalized MPS should have left_errors ≈ 0 everywhere
# and right_errors growing away from zero.  Verify this:

mps_L = FiniteMPS(Spin{1//2}(), L, 32)
left_canonical_sweep!(mps_L)
tensors_L = [t.data for t in mps_L.tensors]

println("Left-canonical MPS:")
errs_L = check_normalization(tensors_L)
#--

# A right-canonical MPS flips the pattern — right_errors ≈ 0, left_errors
# growing.

mps_R = deepcopy(mps_L)
right_canonical_sweep!(mps_R)
tensors_R = [t.data for t in mps_R.tensors]

println("\nRight-canonical MPS:")
errs_R = check_normalization(tensors_R)
#--

# ## Mixed canonical form
#
# After `move_center!(mps, l)`, sites ``1 \ldots l-1`` are left-canonical,
# site ``l`` is unconstrained, and sites ``l+1 \ldots L`` are right-canonical.
# Your function should reflect this pattern clearly.

l = L ÷ 2
mps_M = deepcopy(mps_L)
move_center!(mps_M, l)
tensors_M = [t.data for t in mps_M.tensors]

println("\nMixed canonical (center = site $l):")
check_normalization(tensors_M)
#--

# ## Qritical.jl: reading the canonical form directly
#
# `FiniteMPS` tracks its own form in `mps.form`.  After any sweep or
# `move_center!` call you can query `mps.form` without running any
# extra SVDs:

println("\nCanonical form stored on mps: ", mps_M.form)
println("  → sites 1…$(l-1) left-canonical, sites $(l+1)…$L right-canonical")
#--
