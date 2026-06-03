# # Task 5.4 — Observables III

# !!! question "Task 5.4 — Observables III"
#     Write a function that receives an MPS in ``\Gamma``-``\Lambda`` notation
#     and evaluates the correlation function
#     ``\langle \sigma^z_{L/2}\, \sigma^z_{L/2+r} \rangle``
#     for all distances ``r``.

using LinearAlgebra, Qritical
#--

# ## Long-range correlations via transfer matrices
#
# Inserting two ``\sigma^z`` operators at sites ``l`` and ``l+r`` (``r \ge 1``)
# requires contracting the region between them.  The most efficient approach
# uses the **transfer matrix** of the MPS:
#
# ```math
# T^{[i]}_{\alpha\bar\alpha, \beta\bar\beta}
#   = \sum_\sigma (\Theta_i)^*_{\alpha\sigma\beta}\,(\Theta_i)_{\bar\alpha\sigma\bar\beta}
# ```
#
# For a correlation at distance ``r`` the contraction factorises as:
#
# 1. Build a **left boundary vector** ``v_L`` with the operator ``\sigma^z``
#    inserted at site ``l``:
#    ``(v_L)_{\alpha\bar\alpha} = \sum_{\sigma\sigma'}
#    (\Theta_l)^*_{\alpha\sigma} \sigma^z_{\sigma\sigma'} (\Theta_l)_{\bar\alpha\sigma'}``
#    (virtual legs merged; sum over physical legs).
#
# 2. For each site ``i = l+1, \ldots, l+r-1`` apply the transfer matrix:
#    ``v \leftarrow v \cdot T^{[i]}``.
#
# 3. Close with the operator ``\sigma^z`` at site ``l+r``:
#    ``\langle\sigma^z_l \sigma^z_{l+r}\rangle
#    = \sum_{\alpha\bar\alpha\sigma\sigma'}
#    (v)_{\alpha\bar\alpha}\, (\Theta_{l+r})^*_{\alpha\sigma}\,
#    \sigma^z_{\sigma\sigma'}\, (\Theta_{l+r})_{\bar\alpha\sigma'}``.

σz = [1.0  0.0; 0.0 -1.0]
#--

function _theta(gammas, lambdas, i)
    Λ_L  = i == 1 ? [1.0] : lambdas[i - 1]
    Λ_R  = lambdas[i]
    χL, d, χR = size(gammas[i])
    return gammas[i] .* reshape(Λ_L, χL, 1, 1) .* reshape(Λ_R, 1, 1, χR)
end
#--

function vidal_correlation_lr(gammas, lambdas, op, l)
    L   = length(gammas)
    d   = size(gammas[1], 2)
    Θ_l = _theta(gammas, lambdas, l)   # shape (χL, d, χR_l)
    χL_l, _, χR_l = size(Θ_l)

    ## TODO: Implement the transfer-matrix sweep.
    ##
    ## Step 1 — left boundary with op inserted at l:
    ##   Θ_mat = reshape(Θ_l, χL_l, d, χR_l)
    ##   vL[α,ᾱ] = ∑_{σσ'} Θ_mat[α,σ,β]* * op[σ,σ'] * Θ_mat[ᾱ,σ',β]
    ##   → vL = (conj(Θ_mat_2d))' * (Θ_mat_2d * op')  where Θ_mat_2d = reshape(permutedims(Θ_l,(1,3,2)), χL_l*χR_l, d)
    ##   → vL has shape (χR_l, χR_l) after proper reshape
    ##
    ## Step 2 — propagate through sites l+1, …, l+r-1 via the transfer matrix:
    ##   For each site i: Θ_i = _theta(gammas, lambdas, i)
    ##   T[α,ᾱ,β,β̄] = ∑_σ conj(Θ_i)[α,σ,β] * Θ_i[ᾱ,σ,β̄]
    ##   vL ← reshape(reshape(vL, 1, χR²) * reshape(T, χR², χR'²), χR', χR')
    ##
    ## Step 3 — close with op at l+r:
    ##   corr[r] = ∑_{α,σ,σ'} vL[α,α̅] * Θ_{l+r}[α,σ,β]* * op[σ,σ'] * Θ_{l+r}[α̅,σ',β]

    corrs = zeros(L - l)
    return corrs
end
#--

L = 10; χ = 8; l = L ÷ 2
mps = FiniteMPS(Spin{1//2}(), L, χ)
left_canonical_sweep!(mps)

function to_vidal(tensors, bond_svs)
    gammas  = [tensors[i] ./ reshape(bond_svs[i], size(tensors[i],1), 1, 1) for i in 1:length(tensors)]
    lambdas = [bond_svs[i + 1] for i in 1:length(tensors)]
    return gammas, lambdas
end

gammas, lambdas = to_vidal([t.data for t in mps.tensors], mps.bond_svs)

corrs = vidal_correlation_lr(gammas, lambdas, σz, l)
println("⟨σᶻ_{L/2} σᶻ_{L/2+r}⟩ for r=0,1,…,$(L-l):")
for (r, c) in enumerate(corrs)
    println("  r=$r: $(round(c; sigdigits=4))")
end
#--

# ## Connected correlator and decay
#
# Subtract the product of single-site values to get the connected part
# ``C(r) = \langle\sigma^z_l\sigma^z_{l+r}\rangle
#         - \langle\sigma^z_l\rangle\langle\sigma^z_{l+r}\rangle``.
# For a gapped ground state ``C(r)`` decays exponentially; for a critical
# state it decays as a power law.

sz_all = [let Θ = _theta(gammas, lambdas, i)
              χL, d, χR = size(Θ)
              Θm = reshape(permutedims(Θ,(1,3,2)), χL*χR, d)
              real(sum(conj(Θm) .* (Θm * σz')))
          end for i in 1:L]

corrs_c = corrs .- [sz_all[l] * sz_all[l + r] for r in 1:L-l]
println("\nConnected C(r):")
for (r, c) in enumerate(corrs_c)
    println("  r=$r: $(round(c; sigdigits=4))")
end
#--
