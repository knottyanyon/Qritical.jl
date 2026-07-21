using LinearAlgebra, Serialization, Qritical
const DATA_ROOT = normpath(joinpath(@__FILE__, "..", "..", "..", "data"))
load_mps(fname, D=64) = let
    ψ_raw = deserialize(joinpath(DATA_ROOT, fname))
    N = ndims(ψ_raw);  d = size(ψ_raw, 1)
    sites = Tuple([upper(Symbol(:s, i), d) for i in 1:N])
    to_mps(QTensor(ψ_raw, sites); trunc=MaxBondDimTrunc(D), form=:left)
end
mps1 = load_mps("psi1.jls")
mps2 = load_mps("psi2.jls")
N    = length(mps1.tensors)
println("Loaded two MPS of length ", N)

# # Ex 4. Overlap, Observables, and MPS Addition
#
# ```@raw html
# <details class="page-info-drawer">
#   <summary>
#     <span class="pi-toggle-label">Page info</span>
#     <span class="pi-pills">
#       <span class="status-pill status-draft">draft</span>
#       <span class="status-pill status-needs-rewrite">needs rewrite</span>
#       <span class="status-pill status-needs-proofreading">needs proofreading</span>
#     </span>
#   </summary>
#   <div class="page-info-body">
#     <table class="page-info-table"><tbody>
#       <tr><td class="pi-key">status</td><td class="pi-val"><span class="status-pill status-draft">draft</span> <span class="status-pill status-needs-rewrite">needs rewrite</span> <span class="status-pill status-needs-proofreading">needs proofreading</span></td></tr>
#       <tr><td class="pi-key">last updated</td><td class="pi-val">2026-07-19</td></tr>
#       <tr><td class="pi-key">written by</td><td class="pi-val">Bavithra Govintharajah</td></tr>
#       <tr><td class="pi-key">edited by</td><td class="pi-val">Claude Sonnet 4.6 — initial draft</td></tr>
#     </tbody></table>
#   </div>
# </details>
# ```
#
# **Week 4 — where this sits in the arc.** Weeks 2–3 taught us to *store* a state
# as a canonical MPS. This week we start *asking questions of it*: how much do two
# states overlap, what is the expectation value of a local operator, how do two
# spins correlate, and how do we form a superposition. Every one of these is a
# tensor-network **contraction**, and the recurring miracle is that a canonical
# gauge turns what looks like an $O(d^{N})$ computation into an $O(N\chi^3)$ one.

# ## (a) MPS overlap $\langle \psi_1 | \psi_2 \rangle$
#
# The overlap contracts the bra and ket tensors site by site through a
# $\chi^2$-dimensional transfer matrix — cost $O(L\chi^3 d)$, exponentially
# cheaper than the $O(d^L)$ full-vector dot product.
#
# ```
#   ⟨ψ₁|ψ₂⟩  =   ┌─A─A─A─ ⋯ ─A─┐      bra  (conjugated ψ₁)
#                 │ │ │       │       contract physical legs
#                └─B─B─B─ ⋯ ─B─┘      ket  (ψ₂)
# ```
# Sweeping this "zipper" left to right keeps only a $\chi\times\chi$ transfer
# matrix in memory at any moment — you never build the full $d^{N}$ vector.
#
# !!! info "Why the self-overlap is exactly 1"
#     For a left-canonical MPS, contracting the zipper from the left makes each
#     $A^\dagger A$ pair collapse to the identity ($\sum_\sigma A^{\sigma\dagger}
#     A^{\sigma}=\mathbb{1}$), so $\langle\psi|\psi\rangle$ telescopes down to the
#     norm carried on the far boundary — which is $1$ by construction. That is the
#     canonical form paying off: normalisation is *free*, not a separate integral.

o11 = overlap(mps1, mps1)
o22 = overlap(mps2, mps2)
o12 = overlap(mps1, mps2)
println("⟨ψ₁|ψ₁⟩ = ", round(real(o11); sigdigits=8))
println("⟨ψ₂|ψ₂⟩ = ", round(real(o22); sigdigits=8))
println("⟨ψ₁|ψ₂⟩ = ", round(o12;     sigdigits=6))

# The two assertions below encode physics as guardrails: a normalised state has
# unit self-overlap, and Cauchy–Schwarz caps the cross-overlap of two unit vectors
# at $|\langle\psi_1|\psi_2\rangle|\le 1$. A violation would signal a bug in the
# contraction, not new physics.
# Sanity: self-overlap of a left-canonical MPS is 1 by construction
@assert abs(real(o11) - 1.0) < 1e-12 "‖ψ₁‖ ≠ 1: $o11"
@assert abs(real(o22) - 1.0) < 1e-12 "‖ψ₂‖ ≠ 1: $o22"
# Cauchy-Schwarz: |⟨ψ₁|ψ₂⟩| ≤ 1
@assert abs(o12) ≤ 1.0 + 1e-10  "Cauchy-Schwarz violated"
println("All overlap checks passed ✓")

# ## (b) Local observables $\langle \sigma^z_i \rangle$ and $\langle \sigma^x_i \rangle$
#
# `local_expectation(mps, op, site)` performs a single left-to-right environment
# sweep inserting `op` at `site`.  For a left-canonical MPS the left environment
# collapse is automatic, but the function works on any canonical form.
#
# !!! info "The mixed-canonical shortcut (Schollwöck Eq. 97)"
#     This is the highest-value idea of the week. Put the orthogonality centre
#     *at* site $i$: then every tensor to the left is left-isometric and every
#     tensor to the right is right-isometric, so **both environments collapse to
#     the identity** and the expectation value reduces to a single local trace,
#     ```math
#     \langle O_i\rangle = \operatorname{tr}\!\big(O\,\rho_i\big),\qquad
#     \rho_i = \sum_{\alpha\beta}\Lambda_\alpha M^{\dagger}_{\alpha} M_{\beta}\Lambda_\beta ,
#     ```
#     costing $O(D^2 d^2)$ instead of the $O(L D^3 d)$ of a full sweep. "Measuring
#     on the sweep" — dragging the centre along and reading off observables as you
#     pass each site — is exactly what makes MPS observables cheap.

ops  = algebra_generators(SpinHalf())
σz   = ops.Sz * 2   # σᶻ = 2Sᶻ ∈ {+1, −1}
σx   = (ops.Sp + ops.Sm)   # σˣ = S⁺ + S⁻
σz_vals = [real(local_expectation(mps1, σz, i)) for i in 1:N]
σx_vals = [real(local_expectation(mps1, σx, i)) for i in 1:N]
println("⟨σᶻᵢ⟩: ", round.(σz_vals; sigdigits=3))
println("⟨σˣᵢ⟩: ", round.(σx_vals; sigdigits=3))

# An expectation value is a property of the *state*, so it cannot depend on where
# we chose to park the orthogonality centre. Recomputing $\langle\sigma^z_1\rangle$
# with the centre swept to every bond is therefore both a physics check and a
# demonstration that the mixed-canonical shortcut above agrees with the full sweep.
# Cross-check: put center at each site via BondCanonical and recompute ⟨σᶻ⟩
# The value must be gauge-independent
println("⟨σᶻ₁⟩ computed with center at different bonds:")
for l in 1:N
    mps_c = canonicalize(mps1, BondCanonical(l))
    val   = real(local_expectation(mps_c, σz, 1))
    println("  bond-center=$l: ", round(val; sigdigits=6))
end

# ## (c) Two-site correlations $\langle \sigma^z_i \sigma^z_j \rangle$
#
# `two_site_op(mps, op_i, op_j, i, j)` inserts two operators during a single sweep —
# same $O(L\chi^2 d)$ cost as one local measurement.
# The *connected* correlator subtracts the product of single-site means.
#
# !!! info "Why the connected correlator is the physical one"
#     The raw correlator $\langle\sigma^z_i\sigma^z_j\rangle$ contains a trivial
#     "disconnected" piece $\langle\sigma^z_i\rangle\langle\sigma^z_j\rangle$ that
#     survives even for a product state with no correlations at all. Subtracting it,
#     ```math
#     \langle\sigma^z_i\sigma^z_j\rangle_c
#       = \langle\sigma^z_i\sigma^z_j\rangle - \langle\sigma^z_i\rangle\langle\sigma^z_j\rangle ,
#     ```
#     isolates *genuine* correlation. Its decay with $|i-j|$ is the fingerprint of
#     the phase — exponential in a gapped phase (finite correlation length),
#     algebraic at criticality.

# Nearest-neighbour ⟨σᶻᵢ σᶻᵢ₊₁⟩
zz_nn = [real(two_site_op(mps1, σz, σz, i, i+1)) for i in 1:N-1]
println("⟨σᶻᵢ σᶻᵢ₊₁⟩: ", round.(zz_nn; sigdigits=3))
# Connected correlator ⟨σᶻᵢ σᶻᵢ₊₁⟩_c = ⟨σᶻᵢ σᶻᵢ₊₁⟩ − ⟨σᶻᵢ⟩⟨σᶻᵢ₊₁⟩
zz_c = zz_nn .- σz_vals[1:end-1] .* σz_vals[2:end]
println("Connected: ", round.(zz_c; sigdigits=3))

# ## (d) MPS addition $a|\psi_1\rangle + b|\psi_2\rangle$
#
# `add_mps(a, ψ₁, b, ψ₂)` assembles a block-diagonal MPS of bond dimension
# $\chi_1 + \chi_2$, then recompresses via a left sweep.
# Linearity is verified via the overlap identity
# $\langle\psi_1 | a\psi_1 + b\psi_2\rangle = a\langle\psi_1|\psi_1\rangle + b\langle\psi_1|\psi_2\rangle$.
#
# !!! info "Addition is a direct sum of bond spaces (Schollwöck §4.3)"
#     Superposing two MPS keeps every physical leg but stacks the bond spaces:
#     each bulk tensor becomes block-diagonal, $N^{\sigma_i}=M^{\sigma_i}\oplus
#     \tilde M^{\sigma_i}$, so the bond dimension is *added*, $\chi=\chi_1+\chi_2$.
#     The scalar weights $a,b$ are absorbed into a boundary tensor. Because the two
#     states generally share entanglement structure, that $\chi_1+\chi_2$ is almost
#     always larger than the sum actually needs — which is why a **recompression**
#     sweep follows, trimming the bond back down to its true rank.

a, b = 0.6 + 0.0im, 0.8 + 0.0im
mps_sum = add_mps(a, mps1, b, mps2)
println("Bond dims of sum MPS: ", [size(t.data, 3) for t in mps_sum.tensors])
println("⟨ψ_sum|ψ_sum⟩ = ", round(real(overlap(mps_sum, mps_sum)); sigdigits=8))

# Linearity is the defining property of a superposition, and the overlap identity
# turns it into a number we can check: projecting the sum onto $\langle\psi_1|$
# must return $a\langle\psi_1|\psi_1\rangle + b\langle\psi_1|\psi_2\rangle$ exactly.
# Linearity check: ⟨ψ₁|aψ₁+bψ₂⟩ = a⟨ψ₁|ψ₁⟩ + b⟨ψ₁|ψ₂⟩
expected = a * o11 + b * o12
computed = overlap(mps1, mps_sum)
println("Expected ⟨ψ₁|aψ₁+bψ₂⟩ = ", round(expected; sigdigits=6))
println("Computed              = ", round(computed; sigdigits=6))
println("Error: ", round(abs(expected - computed); sigdigits=3))

# Now we make the recompression *lossy* on purpose: capping the bond at `D_max`
# discards the smallest Schmidt values of the sum. The state fidelity below
# measures how much of the exact superposition survived — near $1$ when the two
# states were nearly collinear or low-rank, smaller when the cut bites.
# Compressed addition: cap bond dimension at D_max
D_max    = 8
mps_comp = add_mps(a, mps1, b, mps2; trunc=MaxBondDimTrunc(D_max))
println("Compressed bond dims (D_max=$D_max): ", [size(t.data, 3) for t in mps_comp.tensors])
# Overlap with exact sum measures truncation fidelity
fidelity = abs(overlap(mps_comp, mps_sum))^2 / (real(overlap(mps_comp, mps_comp)) * real(overlap(mps_sum, mps_sum)))
println("Fidelity |⟨compressed|exact⟩|² = ", round(fidelity; sigdigits=5))
