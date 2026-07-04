# Qritical.jl — TDD Task List, Part 2: Physical Systems & Symmetry (Weeks 6–12+)

> **Companion to `Qritical_MasterPlan_Part2.md`.** This half covers the physics: the DoF/operator layer and MPO (Week 6), time evolution via TEBD (Weeks 7–9), exact diagonalisation (Week 10), variational/VMC (Week 11), and symmetry exploitation (Week 12+). The tensor machinery it builds on (SVD, MPS, canonical forms) is in **Part 1**.


Companion to `Qritical_MasterPlan.md`. Organized **by week = by exercise sheet**, each week carrying both the
bare exercise requirement *and* the architectural extras that make this a reusable library rather than
throwaway scripts. Every component follows the TDD order you prefer:

1. **Physics tests (write first)** — failing tests that encode a mathematical/physical invariant.
2. **Types** — structs/abstract types so tests compile.
3. **Implementation** — until the physics tests pass.
4. **Edge cases** — boundaries, degeneracies, numerical stability.

Convention anchor (assert everywhere): ket/output/outgoing leg → `TIx{Upper}` → **codomain** (`V`,
contravariant); bra/input/incoming leg → `TIx{Lower}` → **domain** (`V'`, covariant); a state's physical
legs are kets (`TIx{Upper}`); contract one `Upper` with one `Lower` (von Delft L5/L8/L10 / MasterPlan §13, §23).
`[~]` marks tasks that *adapt/verify* code already implemented under the old plan (the index layer) rather
than build from scratch.

> **Integration note (2026-06-26):** `src/indices.jl`, `src/qtensor.jl`, and the rewritten `src/svd.jl` are active in the module. `Partition`, `Bipartition`, `complement`, `bipartition`, and `group_legs` are implemented and exported; the SVD layer (`AbstractTrunc`/`NoTrunc`/`MaxBondDimTrunc`/`ValCutoffTrunc`, `FullSVD`/`ReducedSVD`, `do_svd`) is active and passing tests, superseding the old `tensor_svd.jl`. Files `tensor_core.jl`, `tensor_svd.jl` (superseded), `dof.jl`, `finite_mps.jl`, `finite_mpo.jl`, `tebd.jl`, `observables.jl`, and `ed.jl` are written and were tested in v0.2–v0.7 (343 tests), but are currently commented out pending integration with the new index layer. Their test files live in `src/test_*.jl`. Tasks marked `[x]` are fully implemented and active; tasks marked `[~done]` have a working implementation in the old-layer files that is not yet re-wired to the new index layer.

---

## Week 6 — Ex 6: MPO, expectation of an operator, apply MPO, power method (the physical-model week)

### 6.1 DoF layer + space bridge

> **Status:** `dof.jl` has `AbstractDoF`, `Spin{S}`, `Fermionic` (≈`SpinlessFermion`), `HardCoreBoson`,
> `CompositeDoF`, and `hilbert_space` (≈`local_dim`). Missing: `operators(dof)`, `statistics`,
> `physical_space(dof, sym)`, `Electron`/`Majorana` stubs, `Commuting`/`Anticommuting` types.

Physics tests:
- [ ] `operators(SpinHalf())` obey `[Sˣ,Sʸ]=iSᶻ`, `(S⁺)†=S⁻`, `Sᶻ` eigenvalues `±½`.
- [ ] `local_dim(Spin{1//2})==2`, `Spin{1}==3`, `SpinlessFermion==2`, `Electron==4`.
- [ ] `statistics` is `Commuting` for spins/hardcore-boson, `Anticommuting` for fermions/Majorana.
- [ ] `physical_space(SpinHalf(), NoSymmetry())` has dimension 2 (sectorless now).

Implementation:
- [~done] `AbstractDoF`; `Spin{S}`, `Fermionic`/`SpinlessFermion`, `HardCoreBoson` — structs exist.
- [ ] `operators(dof)` returning operator matrices as `NamedTuple`.
- [ ] `local_dim(dof)` (rename/alias from `hilbert_space`); `statistics(dof)`.
- [ ] `Electron`, `Majorana` stubs; `Commuting`/`Anticommuting` statistics types.
- [ ] `physical_space(dof, sym)` (sectorless `ComplexSpace` for now).

Edge cases:
- [ ] `Spin{1}` 3×3 operators; intra-site `Electron` sign convention pinned (deferred but documented).

### 6.2 Operator / Hamiltonian + constructors + MPO

> **Status:** `finite_mpo.jl` has `heisenberg_mpo` (specific Heisenberg, not general XXZ) and `identity_mpo`.
> The general `Operator`/`Hamiltonian` term-list layer (`LocalTerm`, `BondTerm`, `XXZ`, `Ising`) does not
> exist.

Physics tests:
- [ ] XXZ MPO sandwiched in a known state equals the analytic energy (cross-check vs a tiny ED in Week 10).
- [ ] field sign matches the course `−ΣhᵢSᶻ`.
- [ ] `magnetization_squared` MPO equals `(ΣSᶻ)²` evaluated by brute force on `L=4`.

Implementation:
- [ ] `LocalTerm`/`BondTerm`; `Operator{D,G,LT,BT}`; `const Hamiltonian = Operator`.
- [ ] **(pending §5 redesign — open decision)** `AbstractOperator` supertype with `opclass` traits.
- [ ] `XXZ`/`Ising`/`Heisenberg` named constructors via the Operator layer.
- [ ] observable constructors (`local_op`, `total_magnetization`, `staggered_magnetization`, `two_point`, `magnetization_squared`).
- [ ] `MPO(H)` finite-state-machine build; running-sum (bond-dim-3) form for all-to-all.
- [~done] `heisenberg_mpo` and `identity_mpo` exist for specific Heisenberg case.

Edge cases:
- [ ] per-bond/per-site coupling arrays vs scalars; long-range term widens MPO correctly.

### 6.3 `expect`, `apply_mpo`, operator arithmetic, `PowerMethod`

> **Status:** `expectation_value(mps, mpo)` and `apply(mpo, mps)` are in `finite_mpo.jl`. Operator
> arithmetic (`+`, `*`, `identity_operator`) and `PowerMethod` are not implemented.

Physics tests:
- [~done] `expect(ψ, H)` — `expectation_value` exists.
- [ ] `expect(ψ, identity_operator) ≈ ‖ψ‖²`.
- [~done] `apply_mpo` then renormalize preserves norm — `apply` exists.
- [ ] **power method:** GS energy converges to ED `eigmin` on `L≤8` within tol.

Implementation:
- [~done] `expectation_value(mps, mpo)` and `apply(mpo, mps)` exist.
- [ ] operator `+`, scalar `*`, `identity_operator(g,dof)`.
- [ ] `PowerMethod(shift,tol,maxiter,trunc)`; `solve(H,GroundState(),pm)` iterate-renormalize loop.

Edge cases:
- [ ] degenerate GS (shift to separate); convergence stall → max-iter guard with diagnostic.

---

## Week 7 — Ex 7: two-site gate exponential, even/odd update, Trotter step

### 7.1 Time vocabulary + axis-tagged gate ❌ not implemented

> **Status:** `apply_gate!(mps, gate, bond; trunc)` is in `tebd.jl` and applies a pre-computed 4-tensor gate.
> The typed `TimeAxis`/`Propagator{A}`/`EvolutionProtocol` axis machinery does not exist.

Physics tests:
- [ ] `gate(h,dt,RealTime())` is unitary (`G†G ≈ I`); `gate(h,dt,ImaginaryTime())` is Hermitian-PSD.
- [ ] **axis is carried, not lost:** `gate(h, p::EvolutionProtocol{A})` returns a `Propagator{A}`.
- [ ] **unitarity derived from axis:** `opclass(::Propagator{RealTime}) === Unitary()`.
- [ ] **step rule forks structurally:** `step!(ψ, ::Propagator{RealTime})` leaves norm unchanged; `step!(ψ, ::Propagator{ImaginaryTime})` renormalizes.
- [ ] `exp(-iHΔt)` on a 2-site analytic example matches the closed form.

Implementation:
- [ ] `TimeAxis` (`RealTime`/`ImaginaryTime`); `abstract type EvolutionProtocol{A<:TimeAxis}`.
- [ ] `ConstantProtocol{A}` with `dt`, `nsteps`, `total_time`, `hamiltonian_at`.
- [ ] `Propagator{A<:TimeAxis,T}` carrying `data` + `dt`; `_phase`, `opclass`.
- [ ] `gate(h,dt,::A)` and `gate(h, p::EvolutionProtocol{A})`; `step!` forking on axis.
- [ ] bond-Hamiltonian extraction `h⁽²⁾` from the `Operator` bond terms.
- [~done] `apply_gate!(mps, gate, bond; trunc)` exists for raw 4-tensor gates.

Edge cases:
- [ ] non-commuting on-site + bond pieces; exhaustive dispatch over `TimeAxis`.

### 7.2 Even/odd update + Trotter step

> **Status:** `trotter_step!(mps, H_bonds, dt; trunc)` is in `tebd.jl`. The `ProductFormula`/`SuzukiTrotter`
> abstraction and `trotter_steps` do not exist.

Physics tests:
- [ ] applying a gate then its inverse returns the original MPS (overlap `≈ 1`).
- [~done] a single 1st-order Trotter step works — `trotter_step!` exists.
- [ ] 2nd/4th-order steps show the expected error scaling `O(Δt³)`/`O(Δt⁵)` per step.

Implementation:
- [ ] `abstract type ProductFormula` with `SuzukiTrotter(order)`; `trotter_steps(::SuzukiTrotter, dt)`.
- [~done] two-site update: merge → apply gate → `do_svd` split + truncate — done inside `apply_gate!`.

Edge cases:
- [ ] boundary bonds in the even/odd pattern; odd `L`.

---

## Week 8 — Ex 8: full real-time TEBD of the Néel state (the TEBD crown)

### 8.1 `TEBD` + `Quench` + `solve`

> **Status:** `time_evolve(mps, H_bonds, dt, nsteps; trunc)` is in `tebd.jl` and works end-to-end.
> The typed `TEBD`/`Quench`/`solve` interface, `neel_state`, and the `Tracker` do not exist.

Physics tests:
- [ ] total `⟨Sᶻ_tot⟩` conserved along XXZ evolution (it commutes with H).
- [~done] short-time `⟨Sᶻ_i⟩(t)` evolution works — `time_evolve` runs.
- [ ] energy conserved in real time within truncation error — not explicitly asserted.

Implementation:
- [ ] `TEBD(order,trunc;picture)`; `Quench{Ψ}`; `solve(H,Quench,TEBD,ConstantProtocol{RealTime},tracker)`.
- [~done] `time_evolve` provides the loop; wrapping into `solve` is pending.
- [ ] `neel_state(g)` product-state MPS constructor.

Edge cases:
- [ ] entanglement growth forcing bond saturation at `D` — `ϵ(t)` rises, flagged in tracker.

### 8.2 `Tracker` (operator vs structural) ❌ not implemented

Physics tests:
- [ ] `measure(total_magnetization)` (operator) reproduces the direct `expect`.
- [ ] `measure_entropy(center)` on the initial Néel product state `= 0`, then grows post-quench.

Implementation:
- [ ] `NoTracker` null-object; `Tracker(measures; every)`; `measure(O::Operator)`.
- [ ] structural `measure_entropy`/`measure_bonddim`/`measure_trunc`.

Edge cases:
- [ ] `every>1` sampling; entropy free only at the canonical center (else flagged `O(χ³)` recompute).

---

## Week 9 — Ex 9: MBL disorder dynamics + imaginary-time ground state ❌ not implemented

### 9.1 Disorder + imaginary-time GS

Physics tests:
- [ ] fixed `rng` → identical disorder realization (reproducibility).
- [ ] imaginary-time evolution of Néel converges to the ED ground-state energy as `τ→∞` (`L≤10`).
- [ ] increasing disorder `h` slows entanglement growth (MBL signature).
- [ ] `⟨S₁ᶻSᵢᶻ⟩` of the converged GS matches ED.

Implementation:
- [ ] `disorder_realization(n,dist,rng)`; field array fed to `XXZ`.
- [ ] `solve(H,GroundState(),TEBD,ConstantProtocol{ImaginaryTime})` — shares TEBD engine via axis dispatch.
- [ ] `J`-sweep driver.

Edge cases:
- [ ] convergence criterion on energy change; `Δτ` step-size vs bond-dim convergence study.

---

## Week 10 — Ex 10: exact diagonalization (the dense/sparse reference)

### 10.1 `ExactDiagonalization`

> **Status:** `ed.jl` has `DenseHamiltonian`, `dense_hamiltonian` (Heisenberg-specific), and `ground_state`
> (sparse Lanczos via KrylovKit). The general `ExactDiagonalization` algorithm type, `solve` dispatch, and
> `Matrix(::Operator)` / `sparse(::Operator)` paths do not exist.

Physics tests:
- [ ] `Matrix(local_op(SpinHalf(),:Sz,i))` equals the kron-embedded `Sᶻ_i` (compare to hand-built on `L=3`).
- [~done] `ED(:ground)` energy via KrylovKit Lanczos — `ground_state(H)` exists.
- [ ] `eigen(Matrix(H))` eigenvalues are real (Hermitian); spectrum symmetric where the model dictates.

Implementation:
- [ ] `ExactDiagonalization(mode,nev)` / `ED`; `solve(H,GroundState(),ED(:ground))` → `eigsolve(sparse(H))`; `:full` → `eigen(Hermitian(Matrix(H)))`.
- [ ] `Matrix(::Operator)`/`sparse(::Operator)`; `statevector(ψ)` (dense `local_dim^L`).
- [ ] wrap the eigenvalues in an `EigValSpectrum` (§1.4b) — ED is its first real consumer, giving `spectral_gap`/spectrum verbs for free.
- [~done] `dense_hamiltonian` + `ground_state` exist for Heisenberg-specific case.

Edge cases:
- [ ] `local_dim^L` size guard (refuse oversized); `4^L` for `Electron`.

### 10.2 ED time evolution ❌ not implemented

Physics tests:
- [ ] `exp(-iH t)|Ψ⟩` is norm-preserving; matches a fine-`Δt` TEBD run on `L≤10` within truncation error.
- [ ] imaginary-time exact projection converges to the GS (cross-check Week 9).

Implementation:
- [ ] ED propagation under `EvolutionProtocol`/`TimeAxis` sharing the same `Propagator{A}` as TEBD.

Edge cases:
- [ ] long-time stability of dense `exp` vs Krylov; Hermiticity assertion under debug flag.

---

## Week 11 — Ex 11: variational / VMC (analytic + NetKet) — minimal Julia work

### 11.1 What to actually implement

- [ ] (analytic, no code) energy-gradient and QGT derivations — notes only.
- [ ] **ED cross-check** = `solve(H, GroundState(), ED(:ground))` (Week 10) compared to the NetKet result; nothing new.
- [ ] (note) low acceptance = `MetropolisLocal` leaves the total-`Sᶻ` sector → use `MetropolisExchange`; sampler issue, not Qritical.jl.
- [ ] (deferred, not now) `TDVP <: Algorithm` — the MPS realization of the same variational geometry; record as a stub only.

---

## Week 12+ — Symmetry exploitation (the architectural payoff)

### 12.1 Graded spaces + symmetric bond legs ❌ not implemented

Physics tests (write first):
- [ ] `physical_space(SpinlessFermion(), FermionNumber())` is `Rep{U1}(0=>1,1=>1)` of total dim 2.
- [ ] a `U(1)`-charged state has **zero** amplitude in wrong-charge sectors (block structure enforces it).
- [ ] **conservation:** `expect` of a charge-changing operator on a charge eigenstate `= 0` automatically.
- [ ] **parity superselection:** `⟨c⟩ = 0` for any parity eigenstate.
- [ ] **regression:** XXZ GS energy with `U(1)` on equals the sectorless run (same physics, faster).

Implementation:
- [ ] upgrade `physical_space(dof,sym)` to return graded `ElementarySpace`.
- [ ] attach `ElementarySpace` (with sector labels) to virtual-bond `TIx`.
- [ ] flip the active backend to `:tensorkit` via `with_backend`; all operations run unchanged.

Edge cases:
- [ ] bond sector contents updated correctly after truncation; `Lower` leg carries the dual irrep (`Upper` carries the primal).
- [ ] fermionic braiding signs correct — cross-check vs Jordan–Wigner dense route.

### 12.2 Fermions, native vs Jordan–Wigner ❌ not implemented

Physics tests:
- [ ] t–V (native `FermionParity`/`U1`) GS energy equals XXZ-via-JW GS energy (Route A ≡ Route B).
- [ ] Kitaev chain: near-degenerate doublet splitting `~e^{−L}` at `t=Δ, μ=0`.

Implementation:
- [ ] `basis_change(H, target_dof)` (JW spin↔fermion↔Majorana); `tV`/`kitaev_chain`/`majorana_operators`.

Edge cases:
- [ ] long-range fermionic terms: JW string widens the MPO; native grading does not.

---
