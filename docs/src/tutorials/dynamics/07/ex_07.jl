using LinearAlgebra, Qritical
using CairoMakie
L   = 8
dof = SpinHalf()
g   = Chain(L)
H   = XXZ(g; J=1.0, Jz=1.0, h=0.0)   # isotropic Heisenberg
println("XXZ chain  L=$L  bonds=", length(bonds(g)))

# # Ex 7. Trotter–Suzuki Gates and TEBD
#
# **Week 7 — the core of the lecture.** Everything so far was static. This week we
# make the state *move*: solve $|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle$. The full
# propagator $e^{-iHt}$ is a $d^{L}\times d^{L}$ object we cannot form, but for a
# nearest-neighbour Hamiltonian there is a way out. Split it into odd and even bonds,
# $H = H_{\text{odd}} + H_{\text{even}}$, and approximate one small step by a product
# of the two pieces (a **Trotter–Suzuki** decomposition). Within each parity the
# bond terms act on disjoint sites and therefore commute, so a whole parity is a
# layer of independent **two-site gates**. Apply gate → SVD → truncate: that loop is
# **TEBD**, the workhorse of the rest of the course.
#
# Time evolution under a nearest-neighbour Hamiltonian is approximated by a product
# of two-site gates (Trotter decomposition).  Each gate is $U_{ij}(\Delta t) = e^{-i h^{(ij)} \Delta t}$
# for real time and $e^{-h^{(ij)} \tau}$ for imaginary time.
#
# ```
#   even ──┘└──┘└──┘└──   layer 2:  gates on bonds (2,3),(4,5),…
#   odd  ─┐┌─┐┌─┐┌─┐┌──   layer 1:  gates on bonds (1,2),(3,4),…
#          1 2 3 4 5 6    ← the "brick wall" of a single Trotter step
# ```

# ## (a) Local Hamiltonian and gate $e^{-ih^{(2)}\Delta t}$
#
# `bond_hamiltonian(H, b)` extracts the $4\times 4$ two-site matrix for bond $b$.
# `gate(h, dt, RealTime())` exponentiates it.
#
# !!! info "Real-time gates are unitary; imaginary-time gates are not"
#     For **real** time, $U = e^{-i h\,\Delta t}$ with Hermitian $h$ is unitary:
#     $U^\dagger U = \mathbb{1}$, $|\det U| = 1$ — it *rotates* the state and
#     preserves the norm. For **imaginary** time, $U = e^{-h\,\tau}$ is Hermitian
#     positive but **not** unitary: it *shrinks* high-energy components, which is
#     exactly the ground-state projection we exploit in part (c). The unitarity
#     check below is therefore a real-time-only diagnostic.

h12 = bond_hamiltonian(H, 1)   # 4×4 matrix for bond (1,2)
println("h^(1,2) eigenvalues: ", round.(sort(real.(eigvals(h12))); sigdigits=5))
println("Expected (Heisenberg dimer): ", round.([-3/4, 1/4, 1/4, 1/4]; sigdigits=5))

dt  = 0.1
# gate(...) returns a Propagator; its matrix lives in `.data`
U12 = gate(h12, dt, RealTime()).data
println("Gate U = exp(-i h Δt):")
println("  Unitarity ‖U†U − I‖ = ", round(norm(U12' * U12 - I(4)); sigdigits=4))
println("  |det(U)|            = ", round(abs(det(U12)); sigdigits=6))

# All bond gates for the full chain (matrices via `.data`)
gates_rt = [gate(bond_hamiltonian(H, b), dt, RealTime()).data for b in 1:L-1]
println("All $(L-1) bond gates unitary: ",
        all(norm(U' * U - I(4)) < 1e-12 for U in gates_rt) ? "✓" : "✗")

# ## (b) Single odd / even bond sweep via `apply_gate`
#
# Odd bonds $\{(1,2),(3,4),\ldots\}$ and even bonds $\{(2,3),(4,5),\ldots\}$ act on
# non-overlapping sites, so each parity sweep can be applied sequentially.
# `apply_gate(mps, U, bond)` returns a new MPS after SVD-truncation of the two updated sites.
#
# !!! info "Why the odd/even split is the whole trick"
#     All the gates *within one parity* touch disjoint pairs of sites, so they
#     commute and can be applied in any order (or in parallel) with **no** error.
#     The only approximation is splitting $H_{\text{odd}}$ from $H_{\text{even}}$,
#     which do *not* commute: $e^{-iH\Delta t} = e^{-iH_{\text{odd}}\Delta t}
#     e^{-iH_{\text{even}}\Delta t} + O(\Delta t^2\,[H_{\text{odd}},H_{\text{even}}])$.
#     Each `apply_gate` contracts the two-site gate into the MPS, then does one SVD
#     to re-split the pair and truncate the bond that grew — the elementary TEBD move.

# Build a generic initial MPS in left-canonical form.
# apply_gate / trotter_step consume a canonical FiniteMPS (not Vidal Γ-tensors).
mps0 = to_mps(as_state(randn(ComplexF64, 2^L), fill(2, L));
              trunc=MaxBondDimTrunc(32), form=:left)
println("Initial ⟨ψ|ψ⟩ = ", round(real(overlap(mps0, mps0)); sigdigits=8))

# A single parity layer is unitary, so the norm is preserved after the odd sweep…
# Apply odd bonds (1,2), (3,4), … — apply_gate takes the Propagator directly
mps_odd = mps0
for b in 1:2:(L-1)
    G = gate(bond_hamiltonian(H, b), dt, RealTime())
    global mps_odd = apply_gate(mps_odd, G, b; trunc=MaxBondDimTrunc(32))
end
println("After odd sweep  ⟨ψ|ψ⟩ = ", round(real(overlap(mps_odd, mps_odd)); sigdigits=8))

# …and again after the even sweep completes the brick-wall layer.
# Apply even bonds (2,3), (4,5), …
mps_both = mps_odd
for b in 2:2:(L-1)
    G = gate(bond_hamiltonian(H, b), dt, RealTime())
    global mps_both = apply_gate(mps_both, G, b; trunc=MaxBondDimTrunc(32))
end
println("After odd+even sweep ⟨ψ|ψ⟩ = ", round(real(overlap(mps_both, mps_both)); sigdigits=8))

# ## (c) Full Trotter step via `trotter_step`
#
# `trotter_step(mps, H, dt, SuzukiTrotter(1))` applies the first-order decomposition
# (odd then even bonds) in one call.  `SuzukiTrotter(2)` gives the second-order
# palindrome: half-step forward, half-step backward.
#
# !!! info "Order vs error: what you buy with a palindrome"
#     First order applies odd-then-even and carries a per-step error $O(\Delta t^2)$,
#     i.e. $O(\Delta t)$ accumulated over a fixed total time. Second order symmetrises
#     the sequence — $e^{-iH_{\text{odd}}\Delta t/2}\,e^{-iH_{\text{even}}\Delta t}\,
#     e^{-iH_{\text{odd}}\Delta t/2}$ — and the leading error cancels, leaving
#     $O(\Delta t^3)$ per step ($O(\Delta t^2)$ accumulated) for only one extra
#     half-layer. That is almost always the right default; higher (4th) orders exist
#     but cost more layers per step.

# 1st-order Trotter step (canonical MPS in, canonical MPS out)
mps_t1 = trotter_step(mps0, H, dt, SuzukiTrotter(1); trunc=MaxBondDimTrunc(32))
println("1st-order Trotter ⟨ψ|ψ⟩ = ", round(real(overlap(mps_t1, mps_t1)); sigdigits=8))
# 2nd-order Trotter step
mps_t2 = trotter_step(mps0, H, dt, SuzukiTrotter(2); trunc=MaxBondDimTrunc(32))
println("2nd-order Trotter ⟨ψ|ψ⟩ = ", round(real(overlap(mps_t2, mps_t2)); sigdigits=8))

# Exact unitary evolution conserves energy, since $[H, e^{-iHt}] = 0$. So any drift
# $\Delta E$ after a step is purely the Trotter *splitting* error (plus truncation) —
# and the second-order palindrome should show a visibly smaller drift than first order,
# a direct, quantitative readout of the order-vs-error story above.
# Trotter error: real-time evolution conserves energy exactly, so ΔE ≈ 0
# up to the Trotter splitting error (2nd order has smaller ΔE than 1st)
W_xxz = MPO(H)
E0   = real(expect(mps0,   W_xxz))
E_t1 = real(expect(mps_t1, W_xxz))
E_t2 = real(expect(mps_t2, W_xxz))
println("\nEnergy before evolution:        ", round(E0;   sigdigits=8))
println("After 1st-order Trotter step:   ", round(E_t1; sigdigits=8), "  ΔE=", round(E_t1-E0; sigdigits=3))
println("After 2nd-order Trotter step:   ", round(E_t2; sigdigits=8), "  ΔE=", round(E_t2-E0; sigdigits=3))

# !!! note "Imaginary time is a ground-state filter"
#     Rotating $t \to -i\tau$ turns the oscillating $e^{-iE_n t}$ into the decaying
#     $e^{-E_n\tau}$. In the eigenbasis $|\psi(\tau)\rangle \propto \sum_n c_n
#     e^{-E_n\tau}|n\rangle$, every excited amplitude is suppressed relative to the
#     ground state by $e^{-(E_n-E_0)\tau}$, so after enough steps only $|0\rangle$
#     survives. Two practicalities: $e^{-H\tau}$ is non-unitary and shrinks the norm,
#     so we **re-canonicalise** each step and divide $\langle H\rangle$ by
#     $\langle\psi|\psi\rangle$; and starting from the **Néel** state gives a large
#     overlap with the antiferromagnetic ground state, so convergence is fast.

# Imaginary-time Trotter: convergence toward the ground state.
# Start from the Néel state (good overlap with the AFM ground state).
# e^{-Hτ} shrinks the norm, so renormalize (canonicalize) after each step
# and divide ⟨H⟩ by ⟨ψ|ψ⟩ when reading off the energy.
dτ     = 0.05
mps_it = canonicalize(neel_state(g; dof=dof), LeftCanonical())
for step in 1:200
    mps_it = trotter_step(mps_it, H, dτ, SuzukiTrotter(2);
                          axis=ImaginaryTime(), trunc=MaxBondDimTrunc(16))
    global mps_it = canonicalize(mps_it, LeftCanonical())
end
E_it = real(expect(mps_it, W_xxz)) / real(overlap(mps_it, mps_it))
println("Imaginary-time GS energy (L=$L): ", round(E_it; sigdigits=8))

# The imaginary-time TEBD energy should converge to the exact diagonalisation value
# from below-ish (up to Trotter + truncation error); the gap between them is the
# combined algorithmic error and closes as $d\tau\to0$ and the bond cap grows.
# ED reference for comparison
gs = solve(H, GroundState(), ExactDiagonalization(:ground))
println("ED exact GS energy:             ", round(gs.energy; sigdigits=8))
println("TEBD error:                     ", round(abs(E_it - gs.energy); sigdigits=3))
