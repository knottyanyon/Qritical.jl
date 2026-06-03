# # Task 5.2 — Observables I

# !!! question "Task 5.2 — Observables I"
#     Write a function that receives an MPS in ``\Gamma``-``\Lambda`` notation
#     and evaluates expectation values of ``\sigma^z`` and ``\sigma^x``
#     efficiently at all sites.

using LinearAlgebra, Qritical
#--

# ## Single-site observables without environments
#
# In mixed canonical form (Exercise 4) we computed ``\langle O_l \rangle``
# by moving the orthogonality center to site ``l`` — cost ``O(L \chi^3)``.
#
# In Vidal form all sites have the same structure.  Define the
# **doubly-absorbed tensor**:
#
# ```math
# \Theta_i = \Lambda_{i-1}\, \Gamma_i\, \Lambda_i
# ```
#
# Since the environments on both sides evaluate to identity (left-isometry
# of ``\Lambda_{i-1}\Gamma_i`` and right-isometry of ``\Gamma_i\Lambda_i``),
# the expectation value reduces to a single-site trace:
#
# ```math
# \langle O_i \rangle
#   = \sum_{\alpha,\sigma,\sigma',\beta}
#     (\Theta_i)^*_{\alpha\sigma\beta}\, O_{\sigma\sigma'}\, (\Theta_i)_{\alpha\sigma'\beta}
#   = \operatorname{Re}\!\left[\operatorname{tr}\!\left(
#       \tilde\Theta_i^\dagger\, O\, \tilde\Theta_i
#     \right)\right]
# ```
#
# where ``\tilde\Theta_i`` is ``\Theta_i`` reshaped to
# ``(\chi_{i-1}\chi_i) \times d`` (virtual legs merged, physical leg free).
# Cost: ``O(L \chi^2 d^2)`` — no environment sweeps needed.

# Spin-1/2 operators in ``\{|\uparrow\rangle, |\downarrow\rangle\}`` basis:
σz = [1.0  0.0; 0.0 -1.0]
σx = [0.0  1.0; 1.0  0.0]
#--

function vidal_expectation_all(gammas, lambdas, op)
    L  = length(gammas)
    ## TODO: For each site i, compute ⟨O_i⟩.
    ##
    ## For i in 1:L:
    ##   Λ_L    = i == 1 ? [1.0] : lambdas[i - 1]   # Λ_{i-1}
    ##   Λ_R    = lambdas[i]                          # Λ_i
    ##   χL, d, χR = size(gammas[i])
    ##   Θ      = gammas[i] .* reshape(Λ_L, χL,1,1) .* reshape(Λ_R, 1,1,χR)
    ##   Θ_mat  = reshape(permutedims(Θ, (1,3,2)), χL * χR, d)
    ##   vals[i] = real(sum(conj(Θ_mat) .* (Θ_mat * op')))

    vals = zeros(L)
    return vals
end
#--

# Build a left-canonical MPS and convert to Vidal form (Task 5.1 required).

L = 10; χ = 8
mps = FiniteMPS(Spin{1//2}(), L, χ)
left_canonical_sweep!(mps)

function to_vidal(tensors, bond_svs)
    L = length(tensors); T = eltype(tensors[1])
    gammas  = [tensors[i] ./ reshape(bond_svs[i], size(tensors[i],1), 1, 1) for i in 1:L]
    lambdas = [bond_svs[i + 1] for i in 1:L]
    return gammas, lambdas
end

gammas, lambdas = to_vidal([t.data for t in mps.tensors], mps.bond_svs)
#--

σz_vals = vidal_expectation_all(gammas, lambdas, σz)
σx_vals = vidal_expectation_all(gammas, lambdas, σx)

println("⟨σᶻ⟩ at each site: ", round.(σz_vals; sigdigits=4))
println("⟨σˣ⟩ at each site: ", round.(σx_vals; sigdigits=4))
#--

# ## Sanity check: total magnetization must be conserved
#
# ``\sum_i \langle\sigma^z_i\rangle`` is the total ``S^z``.  For a random MPS
# its value is arbitrary, but it must be consistent across different ways of
# computing it.

println("\nTotal ⟨Sᶻ⟩ = ", round(sum(σz_vals) / 2; sigdigits=6))
#--

# ## Cross-check with mixed canonical (Task 4.3)
#
# Moving the center to each site in turn and using the Task 4.3 formula
# must give the same per-site values.

println("\nCross-check with mixed canonical form:")
for l in 1:L
    mps_c = deepcopy(mps)
    move_center!(mps_c, l)
    C      = mps_c.tensors[l].data
    χL, d, χR = size(C)
    Θ_mat  = reshape(permutedims(C, (1,3,2)), χL * χR, d)
    sz_mc  = real(sum(conj(Θ_mat) .* (Θ_mat * σz')))
    println("  site $l: Vidal=$(round(σz_vals[l];sigdigits=4))  MC=$(round(sz_mc;sigdigits=4))")
end
#--
