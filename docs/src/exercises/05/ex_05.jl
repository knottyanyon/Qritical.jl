using LinearAlgebra, Serialization, Qritical
using CairoMakie
ψ_raw  = deserialize(joinpath(DATA_ROOT, "psi.jls"))
N = ndims(ψ_raw);  d = size(ψ_raw, 1)
sites  = Tuple([upper(Symbol(:s, i), d) for i in 1:N])
mps    = to_mps(QTensor(ψ_raw, sites); trunc=MaxBondDimTrunc(32), form=:left)
println("MPS length: ", N, "  max bond dim: ", maximum(size(t.data, 3) for t in mps.tensors))

# # Ex 5. Vidal (Γ-Λ) Notation and Observables
#
# The Vidal form factors each left-canonical tensor $A_i = \Lambda_{i-1}\,\Gamma_i$
# where $\Lambda_k = \mathrm{diag}(\lambda^{[k]}_1,\ldots)$ are the Schmidt values
# at bond $k$.  All physical information is encoded in the $\Gamma\Lambda$ structure.

# ## (a) Convert to Vidal form
#
# `to_vidal(mps)` returns a new `FiniteMPS` in `VidalForm`.
# The round-trip `to_canonical(to_vidal(mps))` should reproduce the original.

mps_v = to_vidal(mps)
println("After to_vidal: ", mps_v.form)
mps_back = to_canonical(mps_v)
println("Round-trip form: ", mps_back.form)
println("Round-trip ⟨ψ|ψ⟩: ", round(real(overlap(mps_back, mps_back)); sigdigits=10))

# Normalization: Σ λᵢ² = 1 at every bond for a normalized MPS
println("Bond norms ‖Λₖ‖² (should be ≈ 1):")
for k in 1:N
    sv = mps_v.bond_svs[k+1].values
    println("  Bond $k: ", round(sum(abs2, sv); sigdigits=6))
end

# ## (b) Entanglement entropy from the Λ spectra; observables on the canonical form
#
# The real payoff of the Vidal form: the bond matrices $\Lambda_k$ **are** the
# Schmidt values across bond $k$, so the bipartite entanglement entropy
# $S_k = -\sum_\alpha \lambda_\alpha^2 \log_2 \lambda_\alpha^2$ is read off directly
# from `mps_v.bond_svs`, with no contraction.
#
# Expectation values, by contrast, are evaluated on the **canonical**
# reconstruction `to_canonical(mps_v)`: Qritical's `local_expectation` /
# `two_site_op` contract the bare site tensors, which reproduce the state only in
# canonical form — not from the bare $\Gamma$ tensors (those carry inverse-$\Lambda$
# factors and would overflow). The round-trip is exact, so these match the
# original `mps`.

ops = algebra_generators(SpinHalf())
σz  = ops.Sz * 2
σx  = ops.Sp + ops.Sm
# Entanglement entropy at each bond, straight from the Vidal Λ Schmidt values
entropy_bits(λ) = let p = abs2.(λ) ./ sum(abs2, λ)
    -sum(pα -> pα > 0 ? pα * log2(pα) : 0.0, p)
end
S_bond = [entropy_bits(mps_v.bond_svs[k+1].values) for k in 1:N-1]
println("Entanglement entropy per bond (bits):")
for k in 1:N-1
    println("  bond $k|$(k+1):  S = ", round(S_bond[k]; sigdigits=4))
end
# Observables on the canonical reconstruction (normalized by ⟨ψ|ψ⟩)
mc   = to_canonical(mps_v)
nrm2 = real(overlap(mc, mc))
σz_site = [real(local_expectation(mc, σz, i)) / nrm2 for i in 1:N]
σx_site = [real(local_expectation(mc, σx, i)) / nrm2 for i in 1:N]
println("\n⟨σᶻᵢ⟩ = ", round.(σz_site; sigdigits=3))
println("⟨σˣᵢ⟩ = ", round.(σx_site; sigdigits=3))
# Cross-check: Vidal→canonical reproduces the original MPS observables
σz_orig = [real(local_expectation(mps, σz, i)) / real(overlap(mps, mps)) for i in 1:N]
println("Max |⟨σᶻ⟩_reconstructed − ⟨σᶻ⟩_original| = ",
        round(maximum(abs.(σz_site .- σz_orig)); sigdigits=4))

# ## (c) Nearest-neighbour correlations $\langle \sigma^z_i \sigma^z_{i+1} \rangle$

# Correlations on the canonical reconstruction (normalized)
zz_nn   = [real(two_site_op(mc, σz, σz, i, i+1)) / nrm2 for i in 1:N-1]
zz_c_nn = zz_nn .- σz_site[1:end-1] .* σz_site[2:end]
println("⟨σᶻᵢ σᶻᵢ₊₁⟩:      ", round.(zz_nn;   sigdigits=3))
println("Connected ⟨··⟩_c: ", round.(zz_c_nn; sigdigits=3))

# ## (d) Long-range correlations $\langle \sigma^z_{L/2}\,\sigma^z_{L/2+r} \rangle$
#
# For distances $r \ge 2$ the transfer matrix between the two operator insertions
# must be propagated explicitly. `two_site_op` handles arbitrary $(i,j)$.

l = N ÷ 2
corrs   = [real(two_site_op(mc, σz, σz, l, l+r)) / nrm2 for r in 1:N-l]
corrs_c = corrs .- [σz_site[l] * σz_site[l+r] for r in 1:N-l]
println("⟨σᶻ_{L/2} σᶻ_{L/2+r}⟩  and  connected C(r):")
for r in 1:N-l
    println("  r=$r:  full=$(round(corrs[r]; sigdigits=4))   connected=$(round(corrs_c[r]; sigdigits=4))")
end

fig = Figure(size=(680, 320))
ax  = Axis(fig[1,1];
    title="Spin-spin correlations from $l th site",
    xlabel="distance r",  ylabel="C(r)",
    xticks=1:N-l,
)
lines!(ax, 1:N-l, corrs;   color=:steelblue, label="full ⟨σᶻ_{L/2} σᶻ_{L/2+r}⟩")
lines!(ax, 1:N-l, corrs_c; color=:orangered, linestyle=:dash, label="connected C(r)")
scatter!(ax, 1:N-l, corrs;   color=:steelblue, markersize=8)
scatter!(ax, 1:N-l, corrs_c; color=:orangered, markersize=8)
axislegend(ax)
fig
