# Qritical.jl — TDD Task List, Part 1: Tensor Machinery & Methods (Weeks 1–5)

> **Companion to `Qritical_MasterPlan_Part1.md`.** This half covers the backend-agnostic tensor machinery: SVD/truncation, the index & spectrum layer, MPS construction, canonical forms, Vidal form, and the cross-cutting engineering tasks. The physical-model weeks (MPO, operators, time evolution, ED, VMC, symmetry — Weeks 6–12+) are in **Part 2**.


Companion to `Qritical_MasterPlan.md`. Organized **by week = by exercise sheet**, each week carrying both the
bare exercise requirement *and* the architectural extras that make this a reusable library rather than
throwaway scripts. Every component follows the TDD order you prefer:

1. **Physics tests (write first)** — failing tests that encode a mathematical/physical invariant.
2. **Types** — structs/abstract types so tests compile.
3. **Implementation** — until the physics tests pass.
4. **Edge cases** — boundaries, degeneracies, numerical stability.

Convention anchor (assert everywhere): ket/output/outgoing leg → `TIx{Upper}` → **codomain** (`V`,
contravariant); bra/input/incoming leg → `TIx{Lower}` → **domain** (`V'`, covariant); a state's physical
legs are kets (`TIx{Upper}`, the all-contravariant coefficient tensor); contract one `Upper` with one
`Lower` (von Delft L5/L8/L10 / MasterPlan §13, §23).
`[~]` marks tasks that *adapt/verify* code already implemented under the old plan (the index layer) rather
than build from scratch.

> **Integration note (2026-06-26):** `src/indices.jl`, `src/qtensor.jl`, and the rewritten `src/svd.jl` are active in the module. `Partition`, `Bipartition`, `complement`, `bipartition`, and `group_legs` are implemented and exported; the SVD layer (`AbstractTrunc`/`NoTrunc`/`MaxBondDimTrunc`/`ValCutoffTrunc`, `FullSVD`/`ReducedSVD`, `do_svd`) is active and passing tests, superseding the old `tensor_svd.jl`. Files `tensor_core.jl`, `tensor_svd.jl` (superseded), `dof.jl`, `finite_mps.jl`, `finite_mpo.jl`, `tebd.jl`, `observables.jl`, and `ed.jl` are written and were tested in v0.2–v0.7 (343 tests), but are currently commented out pending integration with the new index layer. Their test files live in `src/test_*.jl`. Tasks marked `[x]` are fully implemented and active; tasks marked `[~done]` have a working implementation in the old-layer files that is not yet re-wired to the new index layer.

---

## SVD & truncation: the backend-agnostic decision (read once, applies from Week 1 on)

**Primitive:** `MatrixAlgebraKit.svd_trunc` / `svd_compact` / `svd_full` / `svd_vals`, with its
`TruncationStrategy`. Rationale: the identical call dispatches on dense `Matrix` (LAPACK,
`SafeDivideAndConquer` default) **and** TensorKit `TensorMap` (block-wise per sector) — TensorKit is built on
MatrixAlgebraKit — so the native↔symmetry switch is free. `svd_trunc` returns `(U, S, Vᴴ, ϵ)` with `ϵ` =
2-norm of discarded singular values, exactly the error you want tracked.

**Truncation adapter** — `AbstractTrunc` is a thin wrapper that lowers to a MatrixAlgebraKit strategy:

| Qritical truncation | MatrixAlgebraKit |
|---|---|
| `MaxBondDimTrunc(max_χ)` | `truncrank(max_χ)` |
| `ValCutoffTrunc(minval)` | `trunctol(; atol = minval)` |
| `NoTrunc` | `notrunc()` |

**Optional native-only fast path:** `do_svd(...; alg = :iterative)` routes to `TSVD.jl` `tsvd(M, k)`
(pure-Julia Lanczos bidiagonalization; fastest for `k ≪ dim`) or `PROPACK.jl`. Guarded to the dense backend
only, with a docstring warning about degeneracy/clustering convergence. **Never** the default; never on the
TensorKit path.

**The interface contract:** Qritical code calls `do_svd(A::QTensor, bp::Bipartition, trunc::AbstractTrunc)`
and nothing else. Inside, `group_legs` matricises, LAPACK noise is stripped, then the truncation strategy
is applied. The return type encodes the guarantee: `NoTrunc` → `FullSVD` (exact); any other strategy →
`ReducedSVD` (approximate, error in `.ε`).

> **Status (2026-06-26):** `src/svd.jl` is active in the module. Truncation types (`NoTrunc`,
> `MaxBondDimTrunc`, `ValCutoffTrunc`), result types (`FullSVD`, `ReducedSVD`), and `do_svd` are
> implemented and passing tests. `KeepRelative`, `TruncationError`, `KeepMachineEps`, and the
> `as_strategy` → MatrixAlgebraKit adapter are deferred.

---

## Week 1 — Ex 1: SVD, Schmidt rank, truncation, contraction (the foundation week)

**Exercise requirement:** SVD a matrix and a state tensor; find Schmidt rank under a cutoff; SVD image
compression; benchmark contractions.
**Architectural additions (your design):** the whole index/leg layer, `QTensor` + backend switch,
`Bipartition`/`group_legs`, the backend-agnostic `do_svd` with error tracking + truncation styles, and
data I/O — so even "just do an SVD" goes through the library's typed machinery.

### 1.1 Index layer ✅ (active in module, 48 tests passing — unchanged)

Physics/semantics tests:
- [x] `dim(upper(:σ,2)) == 2`; `label(lower(:α,4)) == :α`.
- [x] `TIx{Upper}(:α,4) != TIx{Lower}(:α,4)` — variance is semantic, not cosmetic.
- [x] **Convention test:** `which_space(::TIx{Upper}) == :codomain`, `which_space(::TIx{Lower}) == :domain` (implemented as `which_space`, not `domain_codomain`).
- [x] `dim(MulTIx(upper(:α,2), lower(:β,3))) == 6`; empty `MulTIx` → `dim == 1`.
- [x] `uppers_range` / `lowers_range` — default and custom start, variance, labels (bonus; not in original spec).

Implementation:
- [x] `[~]` `AbstractIx` (`dim`, `label`, `which_space` interface); `IxLoc`/`Upper`/`Lower`; `TIx{L}` with `dim>0`.
- [x] `[~]` `upper`/`lower`/`uppers`/`lowers`; `MulTIx` with computed `dim` + autolabel.
- [x] `which_space(::TIx{Upper})=:codomain`, `which_space(::TIx{Lower})=:domain`.
- [x] `uppers_range`/`lowers_range` with `last` and optional `start` parameter.
- [ ] `bond_label` — exists in `tensor_index.jl` (old layer) but not yet ported to `indices.jl`.

Edge cases:
- [x] `TIx` with `dim ≤ 0` throws; `dim==1` boundary bond allowed; `MulTIx` of a single index passes through.

### 1.2 `QTensor` + backend switch ✅ (active in module, `src/qtensor.jl`)

Physics tests:
- [x] round-trip: build from `Array`, read `.data`, values unchanged; `length(t.indices) == ndims(t.data)`.
- [x] order-0 (scalar) tensor: 0 indices, 0-dim array, value accessible via `t.data[]`.
- [ ] `with_backend(:tensorkit) do … end` produces a `TensorMap`-backed tensor with the same index metadata.

Implementation:
- [x] `QTensor{Element,Order,D}`; `AbstractArray` subtype with full delegation (`size`, `getindex`, `setindex!`, `IndexStyle`, `strides`, `unsafe_convert`).
- [x] index metadata preservation: labels, dims, variance stored in `NTuple` and retrievable.
- [x] `TensorOperations.tensorstructure` override so `@tensor` contractions see `AbstractIx` metadata.
- [ ] `with_backend` via `ScopedValue`; `tensorkit_space(::TIx) = ComplexSpace(dim)` (sectorless for now).

Edge cases:
- [x] order-0 (scalar) tensor works correctly.
- [x] mismatched leg size and array size throws `ArgumentError`.
- [x] `QTensor` shares memory with the original array (no copy on construction).
- [ ] backend switch inside a nested scope restores `:native` on exit.

### 1.3 Bipartition & `group_legs` ✅ (active in module, `src/indices.jl` + `src/qtensor.jl`)

Physics tests:
- [x] `group_legs` of a rank-3 tensor with cut `{σ,vL | vR}` reshapes to a matrix of size
  `(dim(σ)*dim(vL)) × dim(vR)`; round-trips back under the inverse.
- [x] left partition legs become rows (first axis); right partition legs become columns (second axis).
- [x] legs in non-natural order: array axes are permuted to match partition order before reshape.
- [x] fused leg metadata: `MulTIx` on each output leg records constituent legs and correct `dim`.

Implementation:
- [x] `Partition = Vector{AbstractIx}` (type alias, in `indices.jl`).
- [x] `Bipartition` struct; constructor rejects overlapping legs with a descriptive `ArgumentError`.
- [x] `complement(p, indices)` — legs not in `p`, in original order (in `indices.jl`).
- [x] `complement(p, A::QTensor)` — forwards to `A.indices` (in `qtensor.jl`).
- [x] `bipartition(left, indices)` and `bipartition(left, A::QTensor)` convenience constructors.
- [x] `group_legs(A, bp)` — locates legs by equality, permutes axes, reshapes to 2D, wraps as `QTensor` with `MulTIx` legs; coverage check included.
- [x] All five exported: `Partition`, `Bipartition`, `complement`, `bipartition`, `group_legs`.

Edge cases:
- [x] a leg in the bipartition not found in tensor → `ArgumentError`.
- [x] a leg of the tensor not covered by the bipartition → `ArgumentError`.
- [x] empty left partition → result has 1 row.
- [x] empty right partition → result has 1 column.
- [ ] permute as a view (no copy) where possible — currently always copies via `permutedims`.

### 1.4 `do_svd` + truncation + `FullSVD` / `ReducedSVD` ✅ (active in module, `src/svd.jl`)

> **Design note (2026-06-26):** result type was split into `FullSVD` (no truncation, exact) and
> `ReducedSVD` (truncated, carries `.r` and `.ε`). The public function is `do_svd`, not `tensor_svd`.
> Numerical noise is stripped in `_compute_svd_factors` before any truncation strategy sees the data,
> cleanly separating LAPACK rounding from deliberate approximation.

Physics tests:
- [x] **`NoTrunc` keeps full Schmidt rank:** `length(F.Σ.data.diag) == min(m, n)`.
- [x] **`MaxBondDimTrunc` keeps at most D values:** `length(F.Σ.data.diag) == D`.
- [x] **`ValCutoffTrunc` keeps values above threshold:** correct rank after cutoff.
- [ ] **reconstruction:** `U·Σ·Vd ≈ A` (no truncation) to machine precision — `@test_broken`.
- [ ] **isometry:** `U†U ≈ I`, `Vd·Vd† ≈ I` — `@test_broken`.
- [ ] **truncation error identity:** `‖A − U·Σ·Vd‖_F ≈ ε` — `@test_broken`.
- [ ] **Schmidt rank:** number of `σ > cutoff` matches analytically known rank — `@test_broken`.
- [ ] **backend parity:** `Array` vs trivial-sector `TensorMap` gives identical `Σ` and `ε`.

Implementation:
- [x] `AbstractTrunc` abstract type.
- [x] `NoTrunc()` singleton — no fields, dispatch encodes strategy.
- [x] `MaxBondDimTrunc(max_χ::Int)` — keep at most `max_χ` singular values.
- [x] `ValCutoffTrunc(minval::Float64)` — keep all `σ > minval`.
- [x] `FullSVD` result struct (U, Σ, Vd) — returned by `do_svd(..., NoTrunc())`.
- [x] `ReducedSVD` result struct (U, Σ, Vd, r, ε) — returned by all truncating strategies.
- [x] `_truncate_singular_values(Σ, trunc)` internal dispatch → `(r, ε)`.
- [x] `_compute_svd_factors` — LAPACK SVD + Golub–Van Loan noise cleaning + slicing.
- [x] `_assemble_qtensors` — attaches bond legs `(:λL, :λR)` with correct `Upper`/`Lower` variance.
- [x] `do_svd(A, bp, ::NoTrunc)` → `FullSVD`.
- [x] `do_svd(A, bp, trunc::AbstractTrunc)` → `ReducedSVD`.
- [ ] `as_strategy(::AbstractTrunc) -> MatrixAlgebraKit.TruncationStrategy` adapter (deferred).
- [ ] optional `alg=:iterative` → `TSVD.tsvd` fast path (deferred).

Edge cases:
- [ ] degenerate σ — truncation keeps both or neither, never splits a degenerate pair — `@test_broken`.
- [x] complex `A` → real `Σ` (passes via `@test_broken` unexpected pass).
- [x] rank-deficient `A` handled via Golub–Van Loan noise cleaning.

### 1.4b Spectrum & orthogonality-centre hierarchy ❌ not implemented (new design, 2026-06-26)

> **Design note:** the values from a decomposition become first-class objects instead of being dug out of
> `Σ.data.diag`. `Bond` is slimmed to pure geometry (two faces of one link); truncation error moves onto the
> spectrum. The orthogonality centre is leg-referenced and typed (bond vs site) so canonical-form
> conversions are explicit and directional. Whole hierarchy defined up front (per design decision); some
> subtypes are lightly used until later weeks.

Physics tests:
- [ ] `SingValSpectrum` from a matrix SVD: `values` descending, `ε` = 2-norm of dropped σ, `normalized` flag honoured.
- [ ] entanglement entropy on a `SchmidtSpectrum`: product state → `0`; Bell pair → `log₂2 = 1` bit (derived, not stored).
- [ ] entanglement spectrum `= −2 log σᵢ`; von Neumann `= −Σσᵢ² log σᵢ²` with `0 log 0 = 0` guard.
- [ ] `SchmidtSpectrum.center.bond` legs **are** the `Σ` bond legs of the `do_svd` that produced it (same `TIx`, no matching).
- [ ] reabsorption `U·Diagonal(s.values)` and `Diagonal(s.values)·Vᴴ` equal the dense `U·Σ`/`Σ·Vᴴ` (the hot-path scaling).
- [ ] a function requiring a `SchmidtSpectrum` rejects a bare `SingValSpectrum` (physical-location type guard).

Implementation:
- [ ] `Bond` slimmed to `(lower::TIx{Lower}, upper::TIx{Upper})` — pure geometry; **remove** old `trunc`/`ε` fields.
- [ ] `abstract type OrthoCenter`; `BondCenter(bond::Bond)` (both faces — arrow reverses on conversion); `SiteCenter(leg::TIx{Upper})` (one physical ket leg).
- [ ] `abstract type AbstractSpectrum`; shared verbs (`entanglement_entropy(;base)`, `entanglement_spectrum`, `schmidt_rank`/`length`, `spectral_gap`).
- [ ] `SingValSpectrum{V}` (`values::V=Vector{Float64}`, `ε`, `normalized`) — pure matrix spectrum, no location.
- [ ] `EigValSpectrum{V}` (`values`; no `ε`) — defined now; first consumer Week-10 ED; density-op/excitation later (§17).
- [ ] `SchmidtSpectrum{V}` wrapping `SingValSpectrum` + `cut::Bipartition` + `center::BondCenter`; forwards numerics to inner spectrum, adds `bipartition(s)`/`center(s)`.
- [ ] storage rule: `values` stored as `Vector`; `Diagonal(s.values)` built at scaling sites; `Σ::QTensor` derived on demand — never store the spectrum twice.
- [ ] `do_svd` results expose a `SingValSpectrum` (the analysis view) alongside the `Σ` factor (the contraction view); a state cut wraps it as a `SchmidtSpectrum`.

Edge cases:
- [ ] empty/boundary bond → spectrum `[1.0]`, `ε=0`, `normalized=true`; zero σ excluded from entropy.
- [ ] `BondCenter` vs `SiteCenter` dispatch is exhaustive over `OrthoCenter`.

### 1.5 Schmidt coefficients, entanglement entropy, data I/O

> **Status:** the spectrum verbs now live on `AbstractSpectrum` (§1.4b). What remains here is the SVD→spectrum
> glue and the data-I/O layer. `entanglement_entropy`/`entanglement_spectrum` exist in `observables.jl`/
> `finite_mps.jl` under old names; `load_array`, `bipartition_matrix`, `as_state` do not exist.

Physics tests:
- [ ] product state → entanglement entropy `0`; maximally entangled 2-qubit → `log₂ 2 = 1` bit (via `SchmidtSpectrum`).
- [ ] `schmidt_values(s)` squared-sum `≈ 1` for a normalized state; entropy uses log₂ by default.
- [ ] `load_array` reads `.jls`/`.txt`/`.npy` to the same array (round-trip a saved fixture).

Implementation:
- [ ] `schmidt_values`/`entanglement_entropy(; base=2)` as `AbstractSpectrum` accessors (see §1.4b) — wire `do_svd` output to them.
- [ ] `load_array` (dispatch on extension; `.jls`→`deserialize` for trusted course data); `bipartition_matrix`, `as_state`.

Edge cases:
- [ ] zero singular values excluded from entropy (`0·log0 = 0`); empty/edge bonds `[1.0]`.

---

## Week 2 — Ex 2: canonical decomposition (left / right / mixed), max-dim truncation

### 2.1 `FiniteMPS` + `AbstractMPSForm` `[~done]` (in `finite_mps.jl`)

Physics tests:
- [x] a freshly built MPS reconstructs the original full state (`contract_all ≈ ψ`) within truncation error.
- [x] `bond_svs[1] == bond_svs[L+1] == [1.0]` (open-chain boundaries).

Implementation:
- [x] `FiniteMPS{D,T}` (`tensors::Vector{QTensor{T,3}}`, `bond_svs`, `form`); `AbstractMPS` base.
- [ ] **(re-wire)** upgrade `bond_svs` from a bare vector of singular values to `Vector{SchmidtSpectrum}` (§1.4b) — each stored bond carries its spectrum + `cut` + `BondCenter`; boundary bonds hold the trivial `[1.0]` spectrum.
- [x] `AbstractMPSForm`: `CanonicalForm(llim,rlim)`, `VidalForm()`, `ArbitraryForm()`.

Edge cases:
- [ ] `L=1` (no bonds); `L=2` (single bond) — not explicitly tested.

### 2.2 `to_mps` (full state → MPS via iterated SVD) ❌ not implemented

Physics tests:
- [ ] **left-canonical:** every `A_i` satisfies `A_i† A_i ≈ I`; resulting `form == CanonicalForm(L,L+1)`.
- [ ] **right-canonical:** every `B_i` satisfies `B_i B_i† ≈ I`; `form == CanonicalForm(0,1)`.
- [ ] **mixed at site ℓ:** left of ℓ are A's, right are B's, center holds the norm.
- [ ] with `MaxBondDimTrunc(D)`: max bond ≤ `D`; accumulated `ϵ` matches sum of per-bond discarded weights.

Implementation:
- [ ] `to_mps(ψ, dof; trunc, form=LeftCanonical)` — sweep of `do_svd`, carry Σ·Vd (left) or U·Σ (right).
- [ ] physical legs partitioned by the **SVD bipartition** (state rule, §23), not the operator variance rule.

Edge cases:
- [ ] `D` larger than full rank (no truncation); `D=1` (product-state projection); norm tracked through the carry.

---

## Week 3 — Ex 3: gauge transforms by sweeping, normalization check, JW mapping

### 3.1 `CanonicalizeSweep` iterator

> **Status:** `left_canonical_sweep!`, `right_canonical_sweep!`, and `move_center!` are implemented in
> `finite_mps.jl`, but the `CanonicalizeSweep` iterator type and `CanonicalizeConfig` struct are not.

Physics tests:
- [ ] L→R then R→L sweep returns an MPS representing the **same state** (overlap with original `≈ 1`).
- [~done] moving the center one site preserves the norm and the canonical predicate on the gauged tensors (`move_center!` exists).

Implementation:
- [ ] `CanonicalizeConfig(llim,rlim,trunc)` + named ctors (`LeftCanonical`, `RightCanonical`, `BondCanonical(k)`, `SiteCanonical(k)`) — the configs target a typed `OrthoCenter` (§1.4b): `BondCanonical`/`Left`/`Right` → a `BondCenter`, `SiteCanonical` → a `SiteCenter`. Conversions dispatch on the centre kind; left↔right re-gauges the bond (swaps the `Bond`'s `Upper`/`Lower` faces) rather than rebuilding legs.
- [ ] `CanonicalizeState` + `Base.iterate`; per-step `do_svd` (or `left_orth`/`right_orth` when untruncated); each completed bond updates its `SchmidtSpectrum` in `bond_svs`.
- [~done] sweep functions (`left_canonical_sweep!`, `right_canonical_sweep!`, `move_center!`) exist.

Edge cases:
- [ ] already-canonical input is a near-no-op; truncating sweep accumulates `ϵ` monotonically.

### 3.2 Canonical error + JW (analytic) ❌ not implemented

Physics tests:
- [ ] `canonical_error(A_i)` ≈ 0 for a properly canonical tensor; grows with deliberate perturbation.
- [ ] (paper exercise) JW relations: verify the mapped operators reproduce the XXZ algebra on 2–3 sites numerically.

Implementation:
- [ ] `is_canonical(ψ; tol)`, `canonical_error(ψ) = ‖A†A − I‖`.
- [ ] document JW (`basis_change(_, SpinHalf())`) as the Week-6 fermion bridge; Ex 3.4 itself is pen-and-paper.

---

## Week 4 — Ex 4: overlap, single-site observables at the center, adding MPS

### 4.1 `overlap` and single-site expectation `[~done]` (in `finite_mps.jl` + `observables.jl`)

Physics tests:
- [x] `overlap(ψ,ψ) ≈ ‖ψ‖²`; orthogonal product states → `0`.
- [x] `⟨σᶻ⟩` on `|↑…↑⟩ = +1` per site; `⟨σˣ⟩` on an x-eigenstate `= ±1` (via `local_expectation`).
- [x] center-site shortcut equals the full-network contraction.

Implementation:
- [x] `overlap(ψ,φ)=dot(ψ,φ)` via environment contraction.
- [x] single-site `local_expectation(mps, op, site)` in `observables.jl`.

Edge cases:
- [ ] mismatched lengths/physical dims error clearly; non-canonical input falls back to full contraction.

### 4.2 `add_mps` `[~done]` (as `Base.:+` in `finite_mps.jl`)

Physics tests:
- [x] `add_mps(1,ψ,1,ψ)` represents `2|ψ⟩` (overlap with `ψ` `≈ 2‖ψ‖²`); coefficients respected.
- [x] after recompression, norm/overlap preserved within `ϵ`.

Implementation:
- [x] `Base.:+(ψ, φ)` direct-sum bond legs (block-diagonal), boundary contraction, recompress via sweep.
- [x] `_scale(mps, a)` for scalar multiplication.

Edge cases:
- [ ] `a` or `b` `=0`; adding a state to itself; bond growth then truncation back to `D`.

---

## Week 5 — Ex 5: Vidal Γ-Λ form, observables, correlation functions

### 5.1 Vidal form + all-site observables `[~done]` (in `tebd.jl` + `observables.jl`)

Physics tests:
- [x] `to_vidal`/`to_canonical` round-trip preserves the state (overlap `≈ 1`).
- [x] `Γᵢ Λᵢ` recovers the canonical `B_i`; Λ equals the `values` of the stored `bond_svs[i]` `SchmidtSpectrum`.
- [x] all-site `⟨σᶻ⟩` vector matches per-site contraction.

Implementation:
- [x] `to_vidal`/`to_canonical` in `tebd.jl`; `VidalForm()` tag.
- [x] all-site `local_expectation` loop in `observables.jl`.

Edge cases:
- [ ] zeros in Λ (reduced effective bond) handled without division blow-up.

### 5.2 Correlations

> **Status:** `entanglement_spectrum(mps, bond)` is implemented in `observables.jl` (to be re-expressed as
> `entanglement_spectrum(bond_svs[bond]::SchmidtSpectrum)` via the §1.4b verb). The two-point correlator
> contraction (connecting transfer matrices between sites) is not.

Physics tests:
- [ ] NN `⟨σᶻ_i σᶻ_{i+1}⟩` on Néel `= −1`; on `|↑↑⟩` `= +1`.
- [ ] distance-dependent `⟨σᶻ_{L/2} σᶻ_{L/2+r}⟩` decays as physically expected for the test state.

Implementation:
- [ ] correlator contraction (two operator insertions + connecting transfer matrices); reused by `two_point` (Week 6).

---

## Cross-cutting (do continuously, not a separate week)

- [x] **Convention guard tests** — `which_space` tests in `test_indices.jl` lock the domain/codomain convention.
- [ ] **Backend parity tests**: a representative result computed `:native` vs trivial-sector `:tensorkit` must agree.
- [ ] **`ϵ` accounting**: every truncating operation contributes to a running truncation-error budget surfaced to the user — now sourced from each bond's `SchmidtSpectrum.spectrum.ε` (the per-bond `bond_svs`), since `Bond` no longer carries `ε` (§1.4b).
- [x] **Reproducibility**: `Manifest.toml` committed; `rng` passed explicitly where relevant.
- [~done] **Docs as you go**: exercise pages built with `Literate.jl`/`Documenter.jl` for Ex 01–07.

---

*Sequencing note:* Weeks 1, 6, and 12 are the heavy architectural weeks (math layer, physical-model layer,
symmetry upgrade). The rest reuse what those establish — which is the whole point of building a library
rather than per-exercise scripts. Dependencies follow the MasterPlan §16 order; the SVD/truncation interface
(Week 1) is the single most reused component, so invest in its tests first.

*Next priorities (as of 2026-06-23):*
1. **1.1 cleanup** — port `bond_label` into `indices.jl`.
2. **1.2–1.4 re-wire** — integrate `tensor_core.jl`/`tensor_svd.jl` into the active module using the new index types; rename truncation types to `MaxBondDimTrunc`/`ValCutoffTrunc`.
3. **1.5** — add `get_schmidt_coefficients`, `load_array`, `bipartition_matrix`, `as_state`.
4. **2.2** — implement `to_mps` (the gateway to all MPS exercises).
