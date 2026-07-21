using LinearAlgebra, SparseArrays, Qritical
L   = 8
dof = SpinHalf()
g   = Chain(L)
ops = algebra_generators(dof)
# A generic test state: random statevector → left-canonical MPS (D ≤ 16 for L=8)
ψ_rand = as_state(randn(ComplexF64, 2^L), fill(2, L))
mps    = to_mps(ψ_rand; trunc=MaxBondDimTrunc(16), form=:left)
println("MPS  L=$L  D=", maximum(size(t.data, 3) for t in mps.tensors))

# # Ex 6. Matrix Product Operators (MPOs)
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
# **Week 6 — where this sits in the arc.** Weeks 1–5 were entirely about *states*.
# This week introduces the other half of the machinery: **operators**. A many-body
# Hamiltonian is a sum of up to $O(L)$ terms acting on a $d^{L}$-dimensional space —
# yet, just like a state compresses into an MPS, it compresses into a **matrix
# product operator (MPO)**: a chain of small four-leg tensors. With it, both
# $\langle\psi|H|\psi\rangle$ and $H|\psi\rangle$ become cheap sweeps. This is the
# gateway to every algorithm in Part 2 — the power method here, DMRG and TEBD next.

# ## (a) XXZ MPO via `MPO(XXZ(Chain(L)))`
#
# The `MPO` constructor encodes the Hamiltonian as a finite-state machine (FSM)
# over the auxiliary bond.  Bond dimension is 5 for the XXZ chain:
# `IdL → S⁺ → S⁻ → Sᶻ → IdR`.
#
# !!! info "An MPO is a finite-state machine on the bond"
#     Read the MPO bond index as the *state of an automaton* walking along the
#     chain. It starts in `IdL` (nothing placed yet), may fire one operator of a
#     two-site term — entering an intermediate state like `S⁺` — must *complete*
#     that term on the next site, and then rests in `IdR` (term finished). Encoding
#     this as a matrix makes each site tensor $\hat W^{[i]}$ **lower-triangular with
#     identity in the corners**; that triangular shape is exactly what forces the
#     MPO to represent a *sum* of terms rather than a product. For nearest-neighbour
#     XXZ the automaton needs 5 states, hence bond dimension 5.
#
# !!! note "Coupling-placement convention"
#     A subtlety that only bites for *site-dependent* couplings: the bond
#     coefficient $J_i$ is applied at the step that **completes** the term on bond
#     $(i,i{+}1)$. Schollwöck only shows the uniform case, so this choice must be
#     pinned — which is precisely why the exact-diagonalisation cross-checks below
#     matter.

H_xxz = XXZ(g; J=1.0, Jz=1.0, h=0.0)
W_xxz = MPO(H_xxz)
# MPO tensors are raw order-4 arrays (χ_L, d_out, d_in, χ_R)
println("MPO bond dimensions: ", [size(t, 1) for t in W_xxz.tensors], " → ",
        size(W_xxz.tensors[end], 4))
println("Interior W tensor shape: ", size(W_xxz.tensors[L÷2]))

# The $L=2$ isotropic case is the Heisenberg dimer $H=\vec S_1\!\cdot\!\vec S_2$,
# whose spectrum is textbook: a **singlet** ground state at $-\tfrac34$ and a
# threefold-degenerate **triplet** at $+\tfrac14$. Reproducing $\{-0.75, 0.25,
# 0.25, 0.25\}$ from the constructed matrix is the cheapest possible check that the
# operator algebra and coupling convention are right.
# Physical check: L=2 Heisenberg dimer has eigenvalues {−¾, ¼, ¼, ¼}
H2   = XXZ(Chain(2); J=1.0, Jz=1.0, h=0.0)
M2   = matrix_repr(H2)         # 4×4 dense matrix via direct construction
evs2 = sort(real.(eigvals(M2)))
println("L=2 eigenvalues: ", round.(evs2; sigdigits=4))
println("Expected:        [-0.75, 0.25, 0.25, 0.25]")

# A Hamiltonian must be Hermitian, and — being a sum of local terms — its matrix is
# extremely **sparse** (only $O(L\,d^{L})$ of the $d^{2L}$ entries are nonzero).
# Both facts are what make the sparse ED of Week 10 tractable.
# Hermiticity and sparse structure of the full L=6 matrix
H6   = XXZ(Chain(6); J=1.0, Jz=1.0, h=0.0)
M_sp = matrix_repr(H6, SparseFormat())
M_dn = matrix_repr(H6, DenseFormat())
println("Dense: ", size(M_dn), "  Sparse nnz: ", nnz(M_sp), " / ", prod(size(M_sp)))
println("Hermitian: ", norm(M_dn - M_dn') < 1e-13)

# ## (b) All-to-all $\hat S^z_{\mathrm{tot}}^2 = \left(\sum_i S^z_i\right)^2$
#
# The total-$S^z$ squared is itself a `LatticeOperator` built via
# `total_magnetization` (returns $\hat S^z_{\mathrm{tot}}$).  Its square is an
# all-to-all $\sum_{i,j} S^z_i S^z_j$ operator.
#
# !!! info "Why $\hat S^z_{\rm tot}$ is a cheap MPO but its square is not"
#     A single sum $\sum_i S^z_i$ is a *completed-once* automaton — bond dimension
#     2 (before/after placing the one operator). Its square expands to
#     ```math
#     \big(\textstyle\sum_i S^z_i\big)^2 = \sum_i (S^z_i)^2 + 2\!\!\sum_{i<j} S^z_i S^z_j ,
#     ```
#     an **all-to-all** coupling: the automaton must remember "one $S^z$ already
#     placed, still open" across *any* distance, which costs an extra persistent
#     bond state. Long-range and all-to-all operators are exactly where MPO bond
#     dimension grows — a useful thing to feel in your hands here.

# Build the staggered and total magnetization via Qritical operators
M_stag = staggered_magnetization(g; dof=dof)   # Σᵢ (−1)ⁱ Sᶻᵢ
M_tot  = total_magnetization(g; dof=dof)         # Σᵢ Sᶻᵢ
W_stag = MPO(M_stag)
W_tot  = MPO(M_tot)
println("Staggered mag MPO bond dim: ", [size(t, 1) for t in W_stag.tensors])
println("Total mag MPO bond dim: ",     [size(t, 1) for t in W_tot.tensors])

# The all-up state is the maximal-$S^z$ eigenstate, so $\langle S^z_{\rm tot}\rangle
# = L/2$ exactly — a closed-form value the MPO must reproduce with no truncation
# error, isolating the operator construction from any state-approximation error.
# For the all-up state |↑↑…↑⟩, Stot = L/2 so ⟨Stot⟩ = L/2.
# |↑⟩ is basis state 1, so the all-up statevector is e₁ (kron index 1).
v_up     = zeros(ComplexF64, 2^L);  v_up[1] = 1.0
all_up_c = to_mps(as_state(v_up, fill(2, L)); form=:left)
Stot_up = real(expect(all_up_c, W_tot))
println("⟨↑↑…↑|Stot|↑↑…↑⟩ = ", round(Stot_up; sigdigits=6), "  (expected ", L/2, ")")

# ## (c) Expectation value $\langle \psi | H | \psi \rangle$ via MPO
#
# `expect(mps, mpo)` sweeps a three-legged environment (bra × MPO × ket) from
# left to right — cost $O(L\chi^2 d^2 \chi_W)$.
#
# ```
#   ┌─ ψ* ─ ψ* ─ ⋯ ─┐      bra layer   (⟨ψ|)
#   │    │     │           physical legs
#   L─── W ── W ── ⋯ ─R    operator layer  (H as MPO)
#   │    │     │           physical legs
#   └─ ψ ── ψ ── ⋯ ─┘      ket layer    (|ψ⟩)
# ```
# This is Schollwöck's $L$–$W$–$R$ sandwich (Fig. 38): a rank-3 environment
# (bra bond × MPO bond × ket bond) is carried across the chain one site at a time.

E_xxz = expect(mps, W_xxz)
println("⟨ψ|H_XXZ|ψ⟩ = ", round(real(E_xxz); sigdigits=8))
# ⟨H⟩ is a physical number, independent of the MPS gauge — recomputing it from a
# right-canonical copy of the same state must give the identical answer.
# ⟨H⟩ is gauge-invariant: recompute from a right-canonical copy of the same state
E_xxz2 = expect(canonicalize(mps, RightCanonical()), W_xxz)
println("Same ⟨H⟩ from right-canonical: ", round(real(E_xxz2); sigdigits=8))

# The identity operator is the degenerate MPO (bond dimension 1): sandwiching it
# must reproduce $\langle\psi|\psi\rangle$ exactly. It is the unit test for the whole
# expect/sandwich contraction machinery.
# Identity MPO: ⟨ψ|I|ψ⟩ = ⟨ψ|ψ⟩
I_op  = identity_operator(g, dof)
W_Id  = MPO(I_op)
E_Id  = expect(mps, W_Id)
norm2 = real(overlap(mps, mps))
println("⟨ψ|I|ψ⟩ = ", round(E_Id;  sigdigits=8))
println("⟨ψ|ψ⟩   = ", round(norm2; sigdigits=8))
@assert abs(E_Id - norm2) < 1e-12 "Identity MPO should equal norm squared"
println("Identity MPO check ✓")

# ## (d) Applying MPO to MPS: $|\phi\rangle = H|\psi\rangle$
#
# `apply_mpo(mpo, mps)` contracts each W tensor with the corresponding site tensor,
# producing a new MPS with expanded bond dimension $\chi_{\mathrm{new}} = \chi \cdot \chi_W$.
# The result can be truncated immediately via the `trunc` keyword.
#
# !!! info "Applying an operator inflates the bond — then you compress"
#     Contracting the MPO into the MPS *multiplies* the bond dimensions,
#     $\chi\to\chi\cdot\chi_W$, because the state now carries both its own
#     entanglement and the operator's internal automaton state. Left unchecked this
#     blows up under repeated application — so every operator application in a real
#     algorithm (the power method here, time evolution later) is followed by a
#     **truncation** back to a manageable $\chi$. `apply_mpo(...; trunc=...)` fuses
#     the two steps.

Hpsi     = apply_mpo(W_xxz, mps)
Hpsi_can = canonicalize(Hpsi, LeftCanonical())
println("‖H|ψ⟩‖² = ", round(real(overlap(Hpsi_can, Hpsi_can)); sigdigits=8))
println("Bond dims of H|ψ⟩: ", [size(t.data,3) for t in Hpsi_can.tensors])

# A Hermitian $H$ gives the identity $\|H|\psi\rangle\|^2 = \langle\psi|H^2|\psi\rangle$.
# Computing the right-hand side by applying $H$ *twice* and overlapping with $\langle\psi|$
# must match the norm on the left — an end-to-end check that `apply_mpo` is exact
# (before truncation) and composes correctly. $\langle H^2\rangle-\langle H\rangle^2$
# is also the energy variance that certifies an eigenstate in the power method.
# Verify: ‖H|ψ⟩‖² = ⟨ψ|H²|ψ⟩  (if H is Hermitian)
norm2_Hpsi = real(overlap(Hpsi_can, Hpsi_can))
HHpsi      = apply_mpo(W_xxz, Hpsi_can)
HHpsi_c    = canonicalize(HHpsi, LeftCanonical())
E_H2_via_apply = real(overlap(mps, HHpsi_c))  # ⟨ψ|H²|ψ⟩
println("‖H|ψ⟩‖² via norm:      ", round(norm2_Hpsi;      sigdigits=8))
println("⟨ψ|H²|ψ⟩ via double-apply: ", round(E_H2_via_apply; sigdigits=8))

# Now apply with a hard bond cap and measure the fidelity lost: this is the exact
# operation the power method iterates, so its truncation error is what limits how
# well repeated $H$-application can project onto the ground state.
# Truncated application
Hpsi_trunc = apply_mpo(W_xxz, mps; trunc=MaxBondDimTrunc(8))
Hpsi_tr_c  = canonicalize(Hpsi_trunc, LeftCanonical())
println("Truncated bond dims (D=8): ", [size(t.data,3) for t in Hpsi_tr_c.tensors])
trunc_fid  = abs(overlap(Hpsi_tr_c, Hpsi_can))^2 /
             (real(overlap(Hpsi_tr_c, Hpsi_tr_c)) * real(overlap(Hpsi_can, Hpsi_can)))
println("Truncation fidelity: ", round(trunc_fid; sigdigits=4))
