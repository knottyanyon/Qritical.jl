# Qritical.jl — Implementation Roadmap

Versions follow [Semantic Versioning](https://semver.org).
`v0.x` is the development series: the public API is unstable and breaking changes are allowed between minors.
`v1.0.0` is the first stable API commitment.

Each minor version represents a new **capability** — something you could not do in the previous minor.
Within every minor, the TDD ordering is fixed:

1. **Physics behavior tests** — write failing tests that encode mathematical invariants and expected physical results
2. **Types** — introduce structs and abstract types; tests should at least compile
3. **Algorithms** — implement functions until physics tests pass
4. **Edge case tests** — boundary conditions, degenerate inputs, numerical stability

---

## v0.1 — Named directional indices and backing tensors

**New capability:** create an `IndexedTensor` with named, directional indices attached to a plain array.

### Physics behavior tests (write first)
- [x] `ndim(TIx{Upper}(:σ, 2)) == 2`
- [x] `TIx{Upper}(:α, 4) ≠ TIx{Lower}(:α, 4)` — index position is semantically meaningful
- [x] `ndim(MultiIx(:αβ, (idx_α, idx_β))) == ndim(idx_α) * ndim(idx_β)`
- [x] `IndexedTensor` round-trip: constructing from an array and reading `.data` gives back the same values
- [x] `length(t.indices) == ndims(t.data)` — index tuple length matches array order

### Types
- [x] `AbstractIndex`
- [x] `IndexLoc`, `Upper`, `Lower`
- [x] `TIx{L <: IndexLoc}`
- [x] `MultiIx`
- [x] `IndexedTensor{Element, Order, D <: AbstractArray{Element,Order}}`

### Algorithms
- [x] `upper(label, ndim)`, `lower(label, ndim)` — single-index constructor helpers
- [x] `uppers(label => ndim, ...)`, `lowers(label => ndim, ...)` — batch constructor helpers

### Edge case tests
- [x] Order-0 `IndexedTensor` (scalar): empty index tuple, 0-dimensional array
- [x] `MultiIx` wrapping a single constituent: `ndim` equals that index's `ndim`
- [x] `TIx` with `ndim = 1` (auxiliary boundary bond of a finite MPS)
- [x] `TIx` with `ndim ≤ 0` throws `ArgumentError`
- [x] `MultiIx` with empty constituents: `ndim == 1` (empty product)

---

## v0.2 — SVD with controllable truncation

**New capability:** decompose an `IndexedTensor` via SVD, choose a truncation strategy, and get back an exact truncation error.

### Physics behavior tests (write first)
- [x] Full SVD: `‖A - U diag(S) Vd‖ < tol` (exact reconstruction when no truncation)
- [x] Singular values are non-negative and sorted descending
- [x] `U†U ≈ I` (left isometry)
- [x] `Vd Vd† ≈ I` (right isometry)
- [x] `KeepFirst(r)`: output has exactly `r` singular values
- [x] `KeepAbove(atol)`: every retained σ satisfies `σ > atol`; every discarded σ satisfies `σ ≤ atol`
- [x] `KeepRelative(rtol)`: every retained σ satisfies `σ / σ_max > rtol`
- [x] Truncation error `ε = ‖discarded singular values‖₂` is returned and numerically correct
- [x] `KeepFirst(r)` with `r ≥ rank(A)`: no truncation, `ε = 0`

### Types
- [x] `label(::AbstractIndex)` — interface method alongside `ndim`; returns the index name as a `Symbol`
- [x] `MultiIx` optional label: `MultiIx(indices)` auto-generates label by concatenating `label.(indices)`; empty tuple → `:scalar`
- [x] `Partition` — an ordered group of `AbstractIndex` objects representing one set of tensor legs; lives in `tensor_index.jl`
- [x] `Bipartition` — an ordered pair of mutually exclusive `Partition`s; left legs → matrix rows, right legs → columns; constructor checks for index overlap
- [x] `AbstractTruncation`, `KeepFirst`, `KeepAbove`, `KeepRelative`, `KeepMachineEps`
- [x] `Bond` — records a contraction: `lower::TIx{Lower}` (from-tensor leg) + `upper::TIx{Upper}` (to-tensor leg), same label and ndim, optional `ε::Real` truncation error. **`BondIndex` was removed**; virtual legs are plain `TIx{Upper/Lower}`.
- [x] `TensorSVD{Element, SingularElement, UOrder, VdOrder, UData, VdData}` — typed result of `tensor_svd`; fields `U`, `Σ`, `Vd`, `ε`, `normalized`. Supports named and positional destructuring.

### Algorithms
- [x] `group_legs(A::IndexedTensor, bp::Bipartition)` — permute + reshape into a 2-leg `IndexedTensor`; each axis tagged with an auto-labeled `MultiIx` over that partition's indices; validates full index coverage
- [x] `complement(p::Partition, A::IndexedTensor)` — `Partition` of all indices in `A` not in `p`
- [x] `bipartition(left::Partition, A::IndexedTensor)` — convenience constructor pairing `left` with `complement(left, A)`
- [x] `tensor_svd(A::IndexedTensor, bp::Bipartition, trunc::AbstractTruncation; normalize=false)` — native backend only; returns `TensorSVD`; `Σ` is `IndexedTensor{real(T),2,Diagonal}` with two distinct bond legs; `normalize=true` divides diagonal by ``\|A\|_F``
- [x] Re-indexing: `U` carries original left `Partition` indices + `TIx{Lower}` bond leg; `Σ` carries `TIx{Upper}` + `TIx{Lower}`; `Vd` carries `TIx{Upper}` bond leg + original right `Partition` indices. Bond labels derived as `Symbol(:χ, MultiIx_autolabel)` — guaranteed distinct from all original partition index labels.

### Edge case tests
- [x] Rank-deficient input: `KeepFirst(r)` with `r > rank(A)` keeps only non-zero singular values
- [x] `1 × N` and `N × 1` bipartitions (all indices on one side)
- [x] Complex-valued tensor: singular values are real
- [x] `KeepAbove(0.0)`: identical to keeping all singular values

---

## v0.3 — `AbstractDoF` hierarchy and `FiniteMPS` with canonical forms

**New capability:** represent a finite quantum state as an MPS, canonicalize it, and compute overlaps and entanglement entropy.

### Physics behavior tests (write first)
- [x] `hilbert_space(Spin{1//2}())` has dimension 2
- [x] `hilbert_space(Fermionic())` has dimension 2
- [x] `hilbert_space(HardCoreBoson())` has dimension 2
- [x] Fresh `FiniteMPS` from random tensors has `form == ArbitraryForm()`
- [x] After full left sweep: `form == CanonicalForm(L, L+1)` and `‖A[i]† A[i] - I‖ < tol` for all `i`
- [x] After full right sweep: `form == CanonicalForm(0, 1)` and `‖B[i] B[i]† - I‖ < tol` for all `i`
- [x] Left-canonical and right-canonical forms represent the same state: `|⟨ψ_L | ψ_R⟩|² ≈ 1`
- [x] `⟨ψ|ψ⟩ = 1` for a normalized MPS
- [x] Moving orthogonality center one step: `llim`/`rlim` changes by ±1, state is preserved
- [x] `bond_svs[1] == bond_svs[L+1] == [1.0]` (trivial boundary)
- [x] Entanglement entropy of a product state (χ = 1 everywhere) is 0 at every bond
- [x] Entropy formula: `S = -∑ λᵢ² log(λᵢ²)` applied to `bond_svs[i]` matches contraction-based result

### Types
- [x] `AbstractDoF`, `Spin{S}`, `Fermionic`, `HardCoreBoson`
- [x] `StateSite{D <: AbstractDoF}`
- [x] `AbstractMPSForm`, `CanonicalForm(llim, rlim)`, `ArbitraryForm()`
- [x] `FiniteMPS{D <: AbstractDoF, T <: Number, RT <: Real}` — three type params; `RT = real(T)` for `bond_svs` element type

### Algorithms
- [x] `hilbert_space(::AbstractDoF)` — returns local Hilbert space dimension (native: plain `Int`; TensorKit: deferred)
- [x] `left_canonical_sweep!(mps)` — updates `form` to `CanonicalForm(L, L+1)`
- [x] `right_canonical_sweep!(mps)` — updates `form` to `CanonicalForm(0, 1)`
- [x] `move_center!(mps, i)` — shifts orthogonality center to site `i`
- [x] `overlap(ψ, φ)` — `⟨ψ|φ⟩` via boundary-to-boundary contraction
- [x] `entanglement_entropy(mps, bond)` — from `bond_svs[bond]`
- [x] `+(ψ, φ)` — MPS addition via block-diagonal tensor concatenation

### Edge case tests
- [x] `L = 1`: canonicalization is trivially the identity; `⟨ψ|ψ⟩` = squared Frobenius norm of single tensor
- [x] Complex MPS: `⟨ψ|ψ⟩` is real and positive
- [x] `CanonicalForm(1, 2)` and `CanonicalForm(L-1, L)` — center at left/right boundary sites
- [x] `KeepFirst(1)` during sweep on an entangled state: state becomes product state, entropy = 0

---

## v0.4 — `FiniteMPO` and expectation values

**New capability:** represent a Hamiltonian as an MPO, apply it to an MPS, and compute `⟨ψ|H|ψ⟩` via environment contraction.

### Physics behavior tests (write first)
- [x] Heisenberg MPO on `L` sites has virtual bond dimension 5
- [x] `⟨ψ|I|ψ⟩ = ⟨ψ|ψ⟩` for the identity MPO
- [x] `⟨↑↓|H_Heis|↑↓⟩` on two sites matches the known exact value `-J/4`
- [x] MPO × MPS produces an MPS with bond dimension `χ_MPS × χ_MPO`
- [x] `‖H|ψ⟩‖² = ⟨ψ|H²|ψ⟩` (self-consistency of apply + norm)
- [x] `⟨ψ|H²|ψ⟩ - ⟨ψ|H|ψ⟩² ≥ 0` (variance is non-negative)
- [x] Power method: `H|ψ⟩ / ‖H|ψ⟩‖` converges to the ground state; energy converges to `E₀`

### Types
- [x] `OperatorSite{D <: AbstractDoF}`
- [x] `FiniteMPO{D <: AbstractDoF, T <: Number}` with `IdL::Int`, `IdR::Int`

### Algorithms
- [x] `heisenberg_mpo(L; J=1.0, T=Float64)` — builds the `L`-site Heisenberg MPO
- [x] `apply(mpo, mps)` — MPO × MPS contraction → MPS
- [x] `left_environment(mps, mpo, i)`, `right_environment(mps, mpo, i)`
- [x] `expectation_value(mps, mpo)` — `⟨ψ|H|ψ⟩` via environment contraction

### Edge case tests
- [x] `L = 2`: MPO has only left-boundary and right-boundary tensors; no interior
- [x] MPO applied to unnormalized MPS: `⟨ψ|H|ψ⟩` scales as `‖ψ‖²`
- [x] `J = 0`: MPO is proportional to identity; `⟨ψ|H|ψ⟩ = 0` for any normalized `ψ`

---

## v0.5 — Vidal form and TEBD time evolution

**New capability:** represent an MPS in Γ-Λ-Γ form, apply two-site gates, and run Trotter time evolution. Covers Ex 7–9 and the final assignment.

### Physics behavior tests (write first)
- [x] Vidal normalization: `‖Λ_{i-1} Γ_i Λ_i‖ = 1` at each site when Λ are the correct singular values
- [x] `CanonicalForm ↔ VidalForm` round-trip: `|⟨ψ_can | ψ_vidal⟩|² ≈ 1`
- [x] Identity two-site gate: state unchanged, bond dimension does not grow, `ε = 0`
- [x] Unitary two-site gate preserves norm
- [x] `CompositeDoF{Spin{1//2}, Spin{1//2}}` has `hilbert_space` of dimension 4
- [x] Single full Trotter step on Néel state: energy change matches first-order perturbation theory in `dt`
- [x] Entropy at the central bond of a Néel state grows under time evolution (entanglement build-up)
- [x] Imaginary time evolution of a random state converges to the ground state energy `E₀`

### Types
- [x] `VidalForm()`
- [x] `CompositeDoF{D1 <: AbstractDoF, D2 <: AbstractDoF} <: AbstractDoF`

### Algorithms
- [x] `to_vidal(mps::FiniteMPS)` → `FiniteMPS` in `VidalForm`
- [x] `to_canonical(mps::FiniteMPS)` → `FiniteMPS` in `CanonicalForm`
- [x] `apply_gate!(mps, gate, (i, i+1))` — two-site gate update in Vidal form via SVD
- [x] `trotter_step!(mps, H_bonds, dt)` — single odd/even bond Trotter step
- [x] `time_evolve(mps, H_bonds, t_end, dt; trunc)` — full real or imaginary time evolution loop

### Edge case tests
- [x] Gate at boundary bonds `(1, 2)` and `(L-1, L)`: trivial Λ = `[1.0]` on the open side
- [x] `dt → 0` limit: a single Trotter step acts as the identity to first order in `dt`
- [x] Bond dimension cap hit during gate: truncation error `ε > 0`, state approximately preserved
- [x] Imaginary time: norm decays per step; re-normalization after each step restores it

---

## v0.6 — Observables

**New capability:** compute single-site and global observables (local `⟨O⟩`, entanglement spectrum, entropy profile) in closed form without re-running SVD.

### Physics behavior tests (write first)
- [x] `⟨Sz⟩` at every site of `|↑↑…↑⟩` equals `+1/2` for `Spin{1//2}`
- [x] `⟨Sz⟩` of the two-site Néel state `|↑↓⟩` alternates `+1/2, -1/2`
- [x] `entanglement_entropy` of a product state is `0` at every bond
- [x] `entanglement_entropy` of the maximally entangled two-site state equals `log(2)`
- [x] `⟨n⟩ ∈ {0, 1}` for every site of a `Fermionic` product state
- [x] Observable computed in mixed canonical form (center at site `i`) agrees with full contraction

### Algorithms
- [x] `local_expectation(mps, op, i)` — single-site operator `⟨ψ|O_i|ψ⟩` using mixed canonical form
- [x] `entanglement_spectrum(mps, bond)` — returns `bond_svs[bond]` directly (no new SVD)
- [x] `entanglement_entropy(mps)` → `Vector` of per-bond entropies across all `L-1` bonds

### Edge case tests
- [x] Observable at `i = 1` or `i = L`: no virtual bond on the open side
- [x] Observable on unnormalized state: result is `⟨ψ|O|ψ⟩ / ⟨ψ|ψ⟩`
- [x] Zero operator: `⟨O⟩ = 0` regardless of state

---

## v0.7 — Exact diagonalization (Ex 10)

**New capability:** build the full `2^L × 2^L` sparse Hamiltonian matrix and find the ground state exactly. Used as a cross-check for MPS results on small systems.

### Physics behavior tests (write first)
- [ ] `DenseHamiltonian` for `L = 2` Heisenberg matches the known 4×4 matrix exactly
- [ ] `size(H.H) == (2^L, 2^L)` for `Spin{1//2}` on `L` sites
- [ ] Ground state energy from ED matches `⟨ψ_GS|H|ψ_GS⟩` from MPS v0.4 (for `L ≤ 10`)
- [ ] Sparse matrix has at most `O(L × 2^L)` non-zero entries (each two-site term contributes a fixed number)
- [ ] Ground state is real for a real Hamiltonian with real starting vector

### Types
- [ ] `DenseHamiltonian{D <: AbstractDoF, T <: Number}` — wraps `SparseMatrixCSC{T, Int}`

### Algorithms
- [ ] `dense_hamiltonian(L, model::AbstractDoF; kwargs...)` — builds sparse matrix by summing two-site terms
- [ ] `ground_state(H::DenseHamiltonian; tol)` — ground state energy and vector via `KrylovKit.eigsolve`

### Edge case tests
- [ ] `L = 1`: trivial single-site Hamiltonian
- [ ] All-zero Hamiltonian: ground state energy is 0, any normalized vector is a ground state
- [ ] `L = 14`: largest system where exact diagonalization is tractable in `Float64`

---

## v1.0 — Stable public API

All Milestone 1–3 components are implemented, tested, and documented.
`v1.0.0` commits to API stability: no breaking changes without a major version bump.

- [ ] Public API surface reviewed: types, constructors, key algorithms all have stable signatures
- [ ] All `jldoctest` blocks pass
- [ ] `Aqua.jl` checks pass (unbound type parameters, stale dependencies, ambiguities)
- [ ] `CHANGELOG.md` written, covering v0.1 through v1.0
- [ ] Docstrings complete for every exported symbol

---

## Future milestones (post-v1.0)

Specced in `CORE_DESIGN_JL.md` but deferred until a specific algorithm requires them.

- **`Lattice{V,E}` + MetaGraphsNext.jl** — topology beyond 1D chains (PEPS, tree tensor networks, arbitrary geometry)
- **`:tensorkit` backend** — block-sparse contractions with symmetry-sector quantum numbers; virtual bond legs (`TIx{Upper/Lower}`) need sector-carrying `ElementarySpace` objects instead of plain `ndim`
- **`TIx` contraction engine** — full `TensorOperations.jl` integration for label-matched Einstein summation
- **`InfiniteMPS`** — VUMPS for translationally invariant infinite systems; `PeriodicVector` storage, `(AL, AR, AC, C)` four-representation canonical form
