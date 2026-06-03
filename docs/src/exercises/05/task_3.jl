# # Task 5.3 — Observables II

# !!! question "Task 5.3 — Observables II"
#     Write a function that receives an MPS in ``\Gamma``-``\Lambda`` notation
#     and evaluates the nearest-neighbour correlation function
#     ``\langle \sigma^z_i \sigma^z_{i+1} \rangle`` efficiently at all sites.

using LinearAlgebra, Qritical
#--

# ## Two-site observables
#
# The two-site analogue of ``\Theta_i`` is the **two-site doubly-absorbed
# tensor**:
#
# ```math
# \Theta_{i,i+1} = \Lambda_{i-1}\, \Gamma_i\, \Lambda_i\, \Gamma_{i+1}\, \Lambda_{i+1}
# ```
#
# Its shape is ``\chi_{i-1} \times d \times d \times \chi_{i+1}`` and it
# already encodes the correct normalisation.  The two-site expectation value is:
#
# ```math
# \langle O_i \otimes O_{i+1} \rangle
#   = \sum_{\alpha,\sigma,\sigma',\tau,\tau',\beta}
#     (\Theta)^*_{\alpha\sigma\tau\beta}\,
#     (O_1)_{\sigma\sigma'}\, (O_2)_{\tau\tau'}\,
#     (\Theta)_{\alpha\sigma'\tau'\beta}
# ```
#
# Reshape ``\Theta`` to ``(\chi_{i-1}\chi_{i+1}) \times (d^2)`` and apply
# ``O_1 \otimes O_2`` (a ``d^2 \times d^2`` matrix) to the physical legs.

σz = [1.0  0.0; 0.0 -1.0]
#--

function vidal_correlation_nn(gammas, lambdas, op1, op2)
    L   = length(gammas)
    d   = size(gammas[1], 2)
    O12 = kron(op1, op2)   # d²×d² operator on the two-site space

    vals = zeros(L - 1)

    ## TODO: For i in 1:(L-1), compute ⟨op1_i ⊗ op2_{i+1}⟩.
    ##
    ## Λ_L      = i == 1 ? [1.0] : lambdas[i - 1]
    ## Λ_mid    = lambdas[i]
    ## Λ_R      = lambdas[i + 1]
    ## χL, _, χM = size(gammas[i])
    ## χM, _, χR = size(gammas[i + 1])
    ##
    ## # Build Θ_{i,i+1} by contracting along the shared bond χM:
    ## Γi_abs   = gammas[i]   .* reshape(Λ_L,   χL,1,1) .* reshape(Λ_mid, 1,1,χM)
    ## Γip1_abs = gammas[i+1] .* reshape(Λ_mid, χM,1,1) .* reshape(Λ_R,   1,1,χR)
    ##
    ## # Contract: Θ[α,σ,τ,β] = ∑_γ Γi_abs[α,σ,γ] * Γip1_abs[γ,τ,β]
    ## Θ = reshape(reshape(Γi_abs, χL*d, χM) * reshape(Γip1_abs, χM, d*χR), χL, d, d, χR)
    ## Θ_mat    = reshape(permutedims(Θ, (1,4,2,3)), χL * χR, d^2)
    ## vals[i]  = real(sum(conj(Θ_mat) .* (Θ_mat * O12')))

    return vals
end
#--

L = 10; χ = 8
mps = FiniteMPS(Spin{1//2}(), L, χ)
left_canonical_sweep!(mps)

function to_vidal(tensors, bond_svs)
    gammas  = [tensors[i] ./ reshape(bond_svs[i], size(tensors[i],1), 1, 1) for i in 1:length(tensors)]
    lambdas = [bond_svs[i + 1] for i in 1:length(tensors)]
    return gammas, lambdas
end

gammas, lambdas = to_vidal([t.data for t in mps.tensors], mps.bond_svs)

zz = vidal_correlation_nn(gammas, lambdas, σz, σz)
println("⟨σᶻᵢ σᶻᵢ₊₁⟩: ", round.(zz; sigdigits=4))
#--

# ## Connected correlator
#
# The **connected** (or truncated) two-point function removes the product
# of single-site expectation values:
#
# ```math
# \langle \sigma^z_i \sigma^z_{i+1} \rangle_c
#   = \langle \sigma^z_i \sigma^z_{i+1} \rangle
#   - \langle \sigma^z_i \rangle \langle \sigma^z_{i+1} \rangle
# ```

function to_vidal_exp(gammas, lambdas, op)
    [let Λ_L = i==1 ? [1.0] : lambdas[i-1]; Λ_R = lambdas[i]
         χL, d, χR = size(gammas[i])
         Θ = gammas[i] .* reshape(Λ_L,χL,1,1) .* reshape(Λ_R,1,1,χR)
         Θm = reshape(permutedims(Θ,(1,3,2)), χL*χR, d)
         real(sum(conj(Θm) .* (Θm * op')))
     end for i in 1:length(gammas)]
end

sz = to_vidal_exp(gammas, lambdas, σz)
zz_c = zz .- sz[1:end-1] .* sz[2:end]
println("\n⟨σᶻᵢ σᶻᵢ₊₁⟩_connected: ", round.(zz_c; sigdigits=4))
#--
