```@meta
EditURL = "task_4.jl"
```

# Task 5.4 — Observables III

!!! question "Task 5.4 — Observables III"
    Write a function that receives an MPS in ``\Gamma``-``\Lambda`` notation
    and evaluates the correlation function
    ``\langle \sigma^z_{L/2}\, \sigma^z_{L/2+r} \rangle``
    for all distances ``r``.

````julia
using LinearAlgebra, Qritical
````

## Long-range correlations via transfer matrices

Inserting two ``\sigma^z`` operators at sites ``l`` and ``l+r`` (``r \ge 1``)
requires contracting the region between them.  The most efficient approach
uses the **transfer matrix** of the MPS:

```math
T^{[i]}_{\alpha\bar\alpha, \beta\bar\beta}
  = \sum_\sigma (\Theta_i)^*_{\alpha\sigma\beta}\,(\Theta_i)_{\bar\alpha\sigma\bar\beta}
```

For a correlation at distance ``r`` the contraction factorises as:

1. Build a **left boundary vector** ``v_L`` with the operator ``\sigma^z``
   inserted at site ``l``:
   ``(v_L)_{\alpha\bar\alpha} = \sum_{\sigma\sigma'}
   (\Theta_l)^*_{\alpha\sigma} \sigma^z_{\sigma\sigma'} (\Theta_l)_{\bar\alpha\sigma'}``
   (virtual legs merged; sum over physical legs).

2. For each site ``i = l+1, \ldots, l+r-1`` apply the transfer matrix:
   ``v \leftarrow v \cdot T^{[i]}``.

3. Close with the operator ``\sigma^z`` at site ``l+r``:
   ``\langle\sigma^z_l \sigma^z_{l+r}\rangle
   = \sum_{\alpha\bar\alpha\sigma\sigma'}
   (v)_{\alpha\bar\alpha}\, (\Theta_{l+r})^*_{\alpha\sigma}\,
   \sigma^z_{\sigma\sigma'}\, (\Theta_{l+r})_{\bar\alpha\sigma'}``.

````julia
σz = [1.0  0.0; 0.0 -1.0]
````

````
2×2 Matrix{Float64}:
 1.0   0.0
 0.0  -1.0
````

````julia
function _theta(gammas, lambdas, i)
    Λ_L  = i == 1 ? [1.0] : lambdas[i - 1]
    Λ_R  = lambdas[i]
    χL, d, χR = size(gammas[i])
    return gammas[i] .* reshape(Λ_L, χL, 1, 1) .* reshape(Λ_R, 1, 1, χR)
end
````

````
_theta (generic function with 1 method)
````

````julia
function vidal_correlation_lr(gammas, lambdas, op, l)
    Θ_l = _theta(gammas, lambdas, l)   # shape (χL, d, χR_l)
    L   = length(gammas)
    χL_l, _, χR_l = size(Θ_l)

    TT = promote_type(eltype(Θ_l), eltype(op))

    # Step 1: build vL[β,β'] = ∑_{α,σ,σ'} Θ_l*[α,σ,β] op[σ,σ'] Θ_l[α,σ',β']
    vL = zeros(TT, χR_l, χR_l)
    for α in 1:χL_l
        M = Θ_l[α, :, :]           # (d, χR_l)
        vL .+= conj(M)' * op * M   # (χR_l, d)@(d,d)@(d,χR_l) = (χR_l, χR_l)
    end

    corrs = zeros(TT, L - l)

    for r in 1:(L - l)
        # Step 2: propagate transfer matrices for sites l+1 … l+r-1
        vL_cur = copy(vL)
        for i in (l + 1):(l + r - 1)
            Θ_i = _theta(gammas, lambdas, i)
            χL_i, _, χR_i = size(Θ_i)
            new_vL = zeros(TT, χR_i, χR_i)
            for β in 1:χL_i, βp in 1:χL_i
                iszero(vL_cur[β, βp]) && continue
                # T_part[γ,γ'] = ∑_σ Θ_i*[β,σ,γ] Θ_i[βp,σ,γ'] = conj(Θ_i[β,:,:])' * Θ_i[βp,:,:]
                new_vL .+= vL_cur[β, βp] * (conj(Θ_i[β, :, :])' * Θ_i[βp, :, :])
            end
            vL_cur = new_vL
        end

        # Step 3: close with op at site l+r; sum over right boundary (Tr over χR)
        Θ_r = _theta(gammas, lambdas, l + r)
        χL_r, _, χR_r = size(Θ_r)
        X_total = zeros(TT, χL_r, χL_r)
        for γ in 1:χR_r
            M = Θ_r[:, :, γ]          # (χL_r, d)
            X_total .+= conj(M) * op * M'
        end
        corrs[r] = real(sum(vL_cur .* X_total))
    end

    return corrs
end
````

````
vidal_correlation_lr (generic function with 1 method)
````

````julia
L = 10; χ = 8; l = L ÷ 2
mps = FiniteMPS(Spin{1//2}(), L, χ)
left_canonical_sweep!(mps)

function to_vidal(tensors, bond_svs)
    L       = length(tensors)
    gammas  = [tensors[i] ./ reshape(bond_svs[i], size(tensors[i],1), 1, 1) for i in 1:L]
    lambdas = [bond_svs[i + 1] for i in 1:L]
    return gammas, lambdas
end

gammas, lambdas = to_vidal([t.data for t in mps.tensors], mps.bond_svs)

corrs = vidal_correlation_lr(gammas, lambdas, σz, l)
println("⟨σᶻ_{L/2} σᶻ_{L/2+r}⟩ for r=0,1,…,$(L-l):")
for (r, c) in enumerate(corrs)
    println("  r=$r: $(round(c; sigdigits=4))")
end
````

````
⟨σᶻ_{L/2} σᶻ_{L/2+r}⟩ for r=0,1,…,5:
  r=1: 1.309e11
  r=2: -5.231e18
  r=3: 1.05e27
  r=4: -5.605999999999999e36
  r=5: 8.268999999999999e36

````

## Connected correlator and decay

Subtract the product of single-site values to get the connected part
``C(r) = \langle\sigma^z_l\sigma^z_{l+r}\rangle
        - \langle\sigma^z_l\rangle\langle\sigma^z_{l+r}\rangle``.
For a gapped ground state ``C(r)`` decays exponentially; for a critical
state it decays as a power law.

````julia
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
````

````

Connected C(r):
  r=1: 3.285e10
  r=2: -5.231e18
  r=3: 1.05e27
  r=4: -5.605999999999999e36
  r=5: 8.268999999999999e36

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

