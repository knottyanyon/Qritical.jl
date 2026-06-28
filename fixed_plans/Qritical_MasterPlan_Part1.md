# Qritical.jl — Master Plan, Part 1: Tensor Machinery & Methods

**Scope:** the backend-agnostic tensor-network apparatus — indices/legs/partitions, `QTensor`, SVD & spectra, MPS & canonical forms, the algorithm shells (TEBD, ED), tracking, and the `solve` entry point. None of this layer needs to know any specific physics. The physical-systems half (geometry, DoF, couplings, Hamiltonians, named models, symmetry, fermions) is **Part 2**.

> Split from the consolidated Master Plan v4 (2026-06-19) for context-window economy; content is verbatim, only reordered so the math/index layer (§23) leads. Section numbers are retained from the master document, so cross-references like “§§5” / “§§13” point into Part 2.


**Project:** Qritical.jl — tensor-network computations in condensed-matter physics
**Status:** Single consolidated specification — Claude Code implementation guide and docstring/documentation source
**Revision:** v4 (master), 2026-06-19 — split into Part 1 (machinery) + Part 2 (physical systems)

> Supersedes Plan v2, the Fermions/Majoranas supplement, and v3. The degrees-of-freedom layer (§3) is
> rebuilt around the correct distinction: a **DoF is what the model defines** at each site (local Hilbert
> space + algebra + intrinsic statistics); a **basis change** is a separate, optional equivalence map
> between DoFs. This replaces the earlier "two-level system + representation" collapse, which could not
> express the Hubbard electron (4D) or spin-1 (3D). Every type/function carries a docstring-ready
> description. Code blocks are skeletons. TensorKit sector spellings are illustrative.

---


## 23. Mathematical-modeling layer: indices, legs & partitions

This layer is the substrate beneath every form (MPS/MPO/dense) and the home of the index machinery carried
over from the earlier core design (already implemented for the native backend, roadmap v0.1–v0.3). Its
purpose is the **physical-model / mathematical-model separation**: the physical model (geometry, DoF,
couplings, operator — §2–§7) says *what the physics is*; this layer says *how it is laid out as indexed
tensors*. Keeping them apart lets the same physics take different tensor representations, and lets tensor
algebra run with no physics attached. The **leg** is the bridge between the two.

### Indices and legs
```julia
abstract type AbstractIndex end                # interface: dim(::AbstractIndex), label(::AbstractIndex)
abstract type IxLoc end
struct Upper <: IxLoc end                   # contravariant — ket/output/outgoing leg (codomain, V)
struct Lower <: IxLoc end                   # covariant     — bra/input/incoming leg (domain, V')
struct TIx{L<:IxLoc} <: AbstractIndex; label::Symbol; dim::Int; end   # dim>0 enforced
upper(label, dim) = TIx{Upper}(label, dim);  lower(label, dim) = TIx{Lower}(label, dim)
bond_label(base::Symbol, site::Int) = Symbol(base, site)          # :χ3, :α2, …

struct MulTIx <: AbstractIndex                # a grouped (fused) leg
    label::Symbol; indices::Tuple{Vararg{AbstractIndex}}
end
dim(g::MulTIx) = prod(dim, g.indices; init = 1)
```
A **leg** carries three things: a `label` (contraction matching, not positional — robust past two legs), an
**orientation → variance** (`Upper`/`Lower`), and a **space** (its `dim`, upgraded below). Variance is set
by leg orientation, and this is the single point of contact with §13's fixed convention:

| Leg orientation | Index | §13 / von Delft | TensorKit | Symmetry space |
|---|---|---|---|---|
| **outward / outgoing** | `TIx{Upper}` | high / ket-side / output | **codomain** | irrep `V` |
| **inward / incoming** | `TIx{Lower}` | low / bra-side / input | **domain** | dual irrep `V'` |

Settled bond convention (derived from von Delft's covariant notation, L10.2–L10.3): a contraction pairs one
upper (contravariant, `V`) with one lower (covariant, `V'`); reading the chain left-to-right as a
composition of maps, each site's **right** virtual leg is its output (`TIx{Upper}`, codomain) and its
**left** virtual leg is its input (`TIx{Lower}`, domain); **physical legs `σ` of a state are kets →
`TIx{Upper}`** (the state coefficient tensor `ψ^{σ₁…σ_L}` is all-contravariant, L10.1). Contraction is
Einstein — one `Upper` pairs with one `Lower` of the same label — which is what makes the convention pay
off. This is **identical** to the von Delft index rule of §13; the index layer is its concrete realization,
not a second convention. (Upper = contravariant = output = codomain = `V`; lower = covariant = input =
domain = `V'`, per L5/L8/L10.)

### Variance vs. partition (the distinction that ties §13 together)
Two orthogonal notions, kept separate:
- **Variance** (`Upper`/`Lower`) is *intrinsic per leg*, from orientation. It fixes `V` vs. `V'`,
  contractibility, and symmetry bookkeeping. It never changes.
- **Partition** is a *per-operation choice*: which legs are grouped as rows (codomain) vs. columns (domain)
  to matricise a valence-`k` tensor for an SVD. This is the `Bipartition`, applied by `group_legs`.
```julia
const Partition = Vector{AbstractIndex}        # an ordered group of legs
struct Bipartition; left::Partition; right::Partition; end   # ctor checks disjoint; group_legs checks cover
group_legs(A, bp::Bipartition) -> QTensor    # permute+reshape to 2 legs: MulTIx(left) ← MulTIx(right)
complement(p::Partition, A); bipartition(left::Partition, A)  # implicit-other-side conveniences
```
This is exactly §13's state-vs-operator rule, now mechanized: for a **state** all physical legs share one
variance (`Upper`, the contravariant ket index of the coefficient tensor), so the codomain/domain split is
the free Schmidt-cut `Bipartition`; for an **operator** the ket (`Upper`) and bra (`Lower`) legs have
opposite variance, so variance pins the partition. `Bond`
records a chosen contraction as *pure geometry* — a two-sided virtual leg, the two faces of one shared link
(it is *not* an `AbstractIndex`, and it no longer carries truncation data: error now lives on the spectrum,
see below):
```julia
struct Bond; lower::TIx{Lower}; upper::TIx{Upper}; end          # the two faces of one shared virtual leg
```
A bond is genuinely two-sided: tensor `ℓ` sees the link as one variance, tensor `ℓ+1` sees its dual. A
left↔right canonical conversion does not move the bond — it **re-gauges** it, flipping the diagram arrow
through the link, i.e. swapping which face is `Upper` (codomain) and which is `Lower` (domain). Keeping both
faces in `Bond` makes that conversion "swap which face points toward the centre" rather than "rebuild the
missing dual leg."

**Orthogonality centre (leg-referenced, typed).** The centre of orthogonality is defined by a *leg*, not a
position — it is wherever the gauge stops being isometric. There are two geometrically distinct kinds
(`MPS_review.pdf`; tensors.net Tut 3–4): a **bond centre** (the orthogonality centre is a *link*, whose link
matrix is diagonal/positive/descending — i.e. it *is* the Schmidt spectrum, the SVD's unique gauge) and a
**site centre** (the centre is a *tensor*; everything left is left-isometric, right is right-isometric, the
centre tensor neither). The location is typed so the two can never be confused, and each points at the real
leg/`Bond` it gauges:
```julia
abstract type OrthoCenter end
struct BondCenter <: OrthoCenter; bond::Bond; end              # centre on a link — carries both faces
struct SiteCenter <: OrthoCenter; leg::TIx{Upper}; end         # centre on a tensor — its single physical (ket) leg
```
The asymmetry is honest: `BondCenter` holds a whole `Bond` (two faces, because the link's arrow reverses
under conversion); `SiteCenter` holds one `TIx{Upper}` (a site has exactly one physical ket leg, no two-sidedness
to preserve). Left/right/mixed canonicalisation is then phrased directionally — "move the centre to a
boundary bond / to site ℓ" — dispatching on `BondCenter` vs `SiteCenter` (the `BondCanonical(k)`/`SiteCanonical(k)`
configs of the canonical-form layer).

**Spectra (the singular/eigenvalue hierarchy).** The values returned by a decomposition are a first-class
object, not something to dig out of `Σ.data.diag`. A shared abstract supertype carries the spectrum verbs
(entropy, gap, rank); subtypes fix what the numbers *are*:
```julia
abstract type AbstractSpectrum end

struct SingValSpectrum{V} <: AbstractSpectrum     # σᵢ ≥ 0 from an SVD — purely a matrix object
    values::V                                     # the kept σᵢ, descending (V defaults to Vector{Float64})
    ε::Float64                                    # discarded weight (2-norm of dropped σ); 0.0 if untruncated
    normalized::Bool                              # rescaled so Σσᵢ² = 1?
end

struct EigValSpectrum{V} <: AbstractSpectrum      # λᵢ from a diagonalisation (Hamiltonian, density op, transfer matrix)
    values::V                                     # possibly signed/complex; no ε (diagonalisation does not truncate)
end

struct SchmidtSpectrum{V} <: AbstractSpectrum     # physically-charged: a state bipartition's spectrum
    spectrum::SingValSpectrum{V}                  # the bare numbers (forwards entropy/rank to this)
    cut::Bipartition                              # algebraic: which legs are rows vs columns
    center::BondCenter                            # physical: the bond this spectrum gauges (center.bond ≡ the SVD bond)
end
```
`SingValSpectrum` is honestly "just for a matrix" (no physical location — a plain matrix has no gauge);
`SchmidtSpectrum` adds the physical meaning a state cut has, and is a **bond object by construction** (its
`center` is a `BondCenter`, never a site). The `center.bond` legs *are* the `Σ` bond legs of the SVD that
produced it, so the spectrum points into its own decomposition with no matching step. Entanglement varies
with the cut, so a function that needs the physical location can demand a `SchmidtSpectrum` and a bare
`SingValSpectrum` will not typecheck. `EigValSpectrum` is defined now (first consumer: Week-10 ED
eigenvalues; later the density-operator/excitation spectra of §17).

**Storage / performance.** `values` is stored as a plain `Vector{Float64}` (the `{V}` parameter keeps it
backend-agnostic). The hot path — reabsorbing the link into a neighbour when moving the centre, `U ← U·Σ`
or `Σ·Vᴴ ← Σ·Vᴴ`, including in iterative-SVD sweeps that carry the bond rather than doing a fresh QR — uses
`LinearAlgebra.Diagonal(s.values)`, a zero-copy `O(χ)` wrapper that dispatches to `lmul!`/`rmul!`
row/column scaling (never a dense `O(χ²)` diagonal matrix). The `Σ::QTensor` (the `Diagonal`-backed,
leg-carrying tensor needed for `@tensor` contraction) is *derived on demand* from the spectrum, not stored
twice — one source of truth, and the hot path is the cheap one. Schmidt/entanglement-spectrum/entropy/rank
are **derived functions** on `AbstractSpectrum`, not stored fields (entanglement spectrum `= −2 log σᵢ`;
von Neumann entropy `= −Σ σᵢ² log σᵢ²` with the `0 log 0 = 0` guard).

### QTensor and the two backends
```julia
struct QTensor{T, N, D<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::D                                    # the numbers
    indices::NTuple{N, AbstractIndex}          # the leg metadata (label, variance, space)
end
```
`D` selects the execution engine without touching algorithm code: `:native` → `Array` (trivial sectors,
`@tensor` works directly — the learning/debug path and the dense/ED backend of §6); `:tensorkit` →
`TensorKit.TensorMap` (block-sparse by sector — the symmetric/fermionic backend of §6, §13). The active
backend is a `ScopedValue` (`with_backend(:tensorkit) do … end`), not a global flag, so two backends can run
in one session and `:native` stays the safe default. `MPS`/`MPO`/gate/observable tensors are all
`QTensor`s; truncation is the shared `AbstractTrunc` hierarchy (`NoTrunc`; `MaxBondDimTrunc(max_χ)` =
max-bond-dimension; `ValCutoffTrunc(minval)` = absolute cutoff), and `do_svd(A, bp, trunc)` returns a result
type that *encodes its own guarantee*: `NoTrunc` → **`FullSVD`** (U, Σ, Vd; reconstruction exact, no error
field), every truncating strategy → **`ReducedSVD`** (U, Σ, Vd, `r`, `ε`) where `r` is the kept rank and `ε`
the 2-norm of the discarded singular values, so `‖A − U·Σ·Vd‖_F = ε`. The split is deliberate: an exact
factorisation has nothing to apologise for, so it carries no `ε`; only an approximation does, and its error
is impossible to forget because it lives in the type. LAPACK rounding noise is stripped (Golub–Van Loan
rank threshold `n·ε_machine·σ₁`) *before* any truncation strategy sees the spectrum, separating numerical
zeros from a deliberate physical cut. Bond legs are named `:λL` (U↔Σ) and `:λR` (Σ↔Vd). The `Σ` factor is
the contraction view (a `Diagonal`-backed `QTensor`); the *analysis* view of the same numbers is the
`SingValSpectrum` (carrying `ε`, `normalized`) introduced above — derived from, not duplicated alongside,
the factor — and across a state bipartition it is wrapped as a `SchmidtSpectrum` (adding the `cut` and the
`BondCenter`).

### Symmetry bookkeeping through legs (the upgrade path)
The single DoF→space bridge is `physical_space(dof, sym)` (§6) — the symmetry-parametrized form of the core
`hilbert_space(dof)` (`Spin{S}` → `Rep{SU2}(S=>1)`, `SpinlessFermion`/`Electron` → `Rep{FermionParity}`/⊠`U(1)`,
`HardCoreBoson` → `ComplexSpace(2)`). Switching symmetry on is one localized change: a leg's stored space
goes from `ComplexSpace(dim)` to a graded `ElementarySpace` whose **sectors are the good quantum numbers**.
Then:
- a `Upper` (codomain, `V`) leg carries the irrep; a `Lower` (domain, `V'`) leg carries its dual;
- an Einstein contraction (`Upper`↔`Lower`) pairs an irrep with its dual ⇒ the quantum number is conserved,
  enforced by block structure with no checking in algorithm code;
- the payoff concentrates in the **virtual bond legs**: once they carry `ElementarySpace` rather than bare
  `dim`, every tensor becomes block-sparse and contractions run within blocks (the large-`χ` speedup, §13).

So the whole symmetry-exploitation plan rides on the leg's `(variance, space)` pair: variance (from
orientation, fixed by §13/von Delft) decides irrep-vs-dual; space (from `physical_space(dof, sym)`) carries
the quantum numbers. *Deferred (the one real work item):* attaching `ElementarySpace` sector data to
virtual-bond `TIx` — a breaking change to the bond index, made once a concrete symmetry group and model are
chosen. Until then bonds carry `dim` and the native backend runs unchanged. This realizes the §13
`CovIndex`/`LegPartition` closure and keeps the von Delft convention exactly as fixed.

---


*Compiled 2026-06-19. References: Schollwöck, "DMRG in the Age of MPS" (`MPS_review.pdf`); Kennes &
Karrasch, Comp. Phys. Comm. 200, 37 (2016); A. Kitaev, Phys.-Usp. 44, 131 (2001); von Delft, *Tensor
Network Basics* lecture notes 1–2 (`01tensornetworksbasics.pdf`, `02tensornetworksbasics.pdf`) — covariant
index/arrow convention (§13, §23); `CORE_DESIGN_assessment.md` and the earlier core design
(`CORE_DESIGN_JL.md`/`_math.md`/`_mps.md`, `ROADMAP.md`) — index/leg/partition layer (§23); exercise sheets
(`exercise_questions.md`, `exercise_questions_SS26.md`, SS24 Ex10/Ex11); NetKet (Ex 11 VMC, Python — out of scope).*

## 0. Philosophy & scope

Build models the way physicists derive them: pick a geometry, place a degree of freedom, specify
interactions, then ask a question of the resulting operator. Four things are kept strictly separate:

1. **The operator** (`Operator`) — a linear operator as a term list; the `Hamiltonian` is one instance (the
   generator), observables (magnetisation, correlators) are others. Knows nothing about temperature or method.
2. **The representation & storage** — the DoF and its statistics, computational form (MPO/matrix),
   symmetry sectors; plus optional basis changes between DoFs.
3. **The question** (`Study`) — ground state? thermal state? quench? correlator?
4. **The method** (`Algorithm`) — DMRG, TEBD, with their own configuration.

**Type-vs-data rule:** make something a *type* when it forks behaviour or enforces validity by dispatch;
make it *data* when it is a value the math consumes. This decides every choice below.

**DoF vs basis change.** The DoF is the physical content the model defines at a site. A basis change is an
exact equivalence transform to a *different* DoF, used only when wanted and only where an isomorphism
exists. A spin-½ chain may be reframed to spinless fermions (Jordan–Wigner); a Hubbard model is *defined*
on electrons (4D) from the start. Both are first-class.

**Scope discipline.** The course needs `Spin{½}` models (Ising/XXZ/Heisenberg) and their fermionic
reframings (t–V, Kitaev) on a 1D open chain, as MPO and dense matrix, plus real/imaginary-time TEBD.
Everything else — `Electron`/Hubbard, `Spin{1}`, registry datastructure, 2D/periodic geometry, finite-T
purification, open/Lindblad, symmetry exploitation, picture-splitting, native fermionic grading — is
designed as *extension hooks*, implemented later. Rule (`CORE_DESIGN_assessment.md`): build the minimal
abstraction the next exercise needs; do not front-load thesis-scope machinery.

Exercise anchors: **Ex 3/4** (Jordan–Wigner / fermion↔spin), **Ex 6** (MPO of XXZ from arrays
`Jᵢ, Jᵢᶻ, hᵢ`; long-range `ΣΣ Sᶻᵢ Sᶻⱼ`), **Ex 10** (exact diagonalisation), **Final** (TEBD of Néel +
domain wall under XXZ).

---

## 1. Architecture

```
                    ── PHYSICAL MODEL ──                    ── MATHEMATICAL MODEL ──
LAYER 1  Geometry        sites, bonds, boundary, connectivity
LAYER 2  DoF             Spin{S} / SpinlessFermion / Electron / HardCoreBoson / Majorana
                         (intrinsic statistics)   — optional: basis_change between DoFs
LAYER 3  Couplings       the actual Jᵢ, hᵢ, Jᵢⱼ arrays (disorder lives here)
                 │                                  legs / indices (TIx{Upper|Lower}),
                 ▼               ── bridged ──▶      partitions (Bipartition), QTensor
          Hamiltonian              a pure linear operator (term list over a DoF)   (§23)
                 │
   ┌─────────────┼───────────────┐
 MPO(H)       Matrix(H)        sparse(H)        ← computational form (QTensors w/ typed legs)
                 │
                 ▼
          solve(H, study, algorithm, protocol, tracker)
                 │
   Study (question) × Algorithm (method) × EvolutionProtocol{TimeAxis} × Tracker
```

**Two layers, bridged by legs.** The **physical model** (geometry, DoF, couplings, operator) says *what the
physics is*; the **mathematical model** (indices, legs, partitions, `QTensor` — §23) says *how it is
represented as indexed tensors*. They are deliberately separated so the same physics can take different
tensor representations, and so tensor algebra can run with no physics attached. The bridge is the **leg**: a
leg carries a label, an orientation (→ index variance `Upper`/`Lower`, §13/§23), and a space (its `dim`,
upgraded to a symmetry-graded `ElementarySpace` carrying good quantum numbers once symmetry is switched on).

**Dependency graph (acyclic).** Geometry → DoF → Couplings → Hamiltonian → forms → {Study, Algorithm,
EvolutionProtocol, Tracker} → `solve`. Only cross-link: Tracker → algorithm step-info. No cycles.

**Two reconciling abstractions.** Basis statistics, symmetry, and storage all reduce to (a) the elementary
space `S` (set by DoF + symmetry; the DoF's statistics fixes whether `S` is fermionic-graded) and (b) each
tensor's domain→codomain partition. Native dense is the trivial-sector instance of the same system, so
symmetry and native fermions are backend swaps, not rewrites (§13, §23).

---

## 6. Representation layer: spaces, forms, backends

### `physical_space`
```julia
physical_space(dof::AbstractDoF, sym::AbstractSymmetry) -> S
```
Return the elementary space `S` for one site. The **single funnel** through which statistics, symmetry,
and storage backend are chosen:
- `statistics(dof) == Commuting`, `NoSymmetry()` → a plain `local_dim`-dimensional space (LinearAlgebra).
- `Commuting`, `U1()`/`SU2()` → an ordinary graded space (TensorKit).
- `statistics(dof) == Anticommuting`, native backend → a **fermionic-graded** (parity) space; signs via
  braiding (§13). Optionally `⊠ U(1)` (charge, and spin for `Electron`) when conserved.
- `Anticommuting`, dense backend → realised via `basis_change(H, Spin{1//2}())` (Jordan–Wigner) on a plain
  space.

Every tensor is built from `S`, so changing `sym` (or the DoF's statistics) propagates through the whole
stack without touching any algorithm. `physical_space` is the symmetry-parametrized form of the core
`hilbert_space(dof)` bridge (`Spin{S}`→`Rep{SU2}`, fermions→`Rep{FermionParity}`(⊠`U(1)`),
`HardCoreBoson`→`ComplexSpace(2)`); it sets the space a physical **leg** carries (§23), and through that leg
the symmetry sectors (good quantum numbers) flow into every form below.

### `AbstractSymmetry` and subtypes
```julia
abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end                    # trivial sectors → dense (course default)
struct U1         <: AbstractSymmetry end                    # abelian Sᶻ / particle number
struct SU2        <: AbstractSymmetry end                    # non-abelian full spin (later)
abstract type AbstractFermionSymmetry <: AbstractSymmetry end
struct FermionParity <: AbstractFermionSymmetry end          # ℤ₂ parity only (pairing models)
struct FermionNumber <: AbstractFermionSymmetry end          # U(1) number ⊠ parity (no pairing)
# Electron may further carry U(1)_charge ⊠ U(1)_spin ⊠ parity.
```
The symmetry to exploit. Fixed once, at space construction, inherited by every derived tensor. The
fermionic variants exist because an `Anticommuting` DoF on the native backend needs at least parity grading
for its signs (§13–14).

### `MPO`, `Matrix`, `sparse`
```julia
MPO(H::Hamiltonian; sym = NoSymmetry(), backend = :native) -> FiniteMPO
Base.Matrix(H::Hamiltonian) -> Matrix                    # dense 2^(…); Ex 10 reference; small systems
SparseArrays.sparse(H::Hamiltonian) -> SparseMatrixCSC   # Lanczos/Krylov; numerical sparsity
```
`MPO` builds the matrix-product operator with the convention domain = `(left ⊗ phys_in)`,
codomain = `(phys_out ⊗ right)` — where `phys_in` is the bra/input leg (`TIx{Lower}`, domain) and
`phys_out` the ket/output leg (`TIx{Upper}`, codomain) — so it is a map `ℋ → ℋ`. If
`statistics(H.dof) == Anticommuting`, `MPO`
either Jordan–Wigners to a spin DoF first (dense) or builds fermionic-graded tensors (native), per
`sym`/`backend`. `Matrix` builds the dense many-body matrix in the computational basis (validation; small
size — note the dimension is `local_dim^L`, e.g. `4^L` for `Electron`). `sparse` builds the CSC matrix for
Krylov ground states. Numerical CSC sparsity ≠ the symmetry-block structure of §13.

---

## 8. Studies — the question

```julia
abstract type Study end
abstract type EquilibriumStudy <: Study end
abstract type DynamicStudy     <: Study end

struct GroundState  <: EquilibriumStudy end
struct ThermalState <: EquilibriumStudy; β::Float64; end                      # deferred
struct Quench{Ψ}    <: DynamicStudy; ψ0::Ψ; t_final::Float64; end
struct Correlator{OA,OB} <: DynamicStudy
    A::OA; iA::Int; B::OB; iB::Int; t_final::Float64; equilibrium::Bool
end
```
A `Study` encodes the question (regime + target), conditions as data; separate from algorithms so the same
question can be answered by different methods. `GroundState` targets the minimum-energy eigenstate
(property of `H` alone; no temperature). `ThermalState` targets the Gibbs mixture `ρ=e^{-βH}/Z` (at T>0
there is no ground state — a Boltzmann mixture over all eigenstates; purification/METTS; *deferred*).
`Quench` evolves `ψ0` and follows it (the Final; observables via `Tracker`). `Correlator` is `⟨A(t)B⟩`, the
study the picture-split (§10) operates on; a quench is its pure-Schrödinger special case; `equilibrium=true`
enables the time-translation factor-of-2 trick. *Extension.*

---

## 9. Algorithms — the method

```julia
abstract type Algorithm end
struct DMRG <: Algorithm; maxdim::Int; tol::Float64; nsweeps::Int; end
struct TEBD <: Algorithm; formula::ProductFormula; trunc::AbstractTrunc; picture::Picture; end
TEBD(order::Int, trunc; picture = Schrödinger()) = TEBD(SuzukiTrotter(order), trunc, picture)

abstract type ProductFormula end                             # the DECOMPOSITION axis: how the propagator splits into gates
struct SuzukiTrotter <: ProductFormula; order::Int; end      # 1, 2, 4, … (even → Suzuki recursion)
trotter_steps(f::SuzukiTrotter, dt) -> Vector{Tuple{Symbol,Float64}}

# Exact diagonalisation — the small-system dense/sparse backend as a first-class algorithm (Ex 10)
struct ExactDiagonalization <: Algorithm
    mode::Symbol      # :ground → sparse Lanczos, few eigenpairs;  :full → dense, all eigenpairs
    nev::Int          # number of eigenpairs when :ground
end
const ED = ExactDiagonalization
ED(; mode = :ground, nev = 1) = ExactDiagonalization(mode, nev)
```
`DMRG` is variational ground-state search (valid for `EquilibriumStudy`). `TEBD` is Trotterised gate
evolution + per-bond truncation (real-time for dynamics, imaginary-time for ground/thermal prep); `picture`
selects how a correlator distributes evolution. `SuzukiTrotter` is the *product-formula decomposition* — a
distinct axis from the time grid (it answers "how is `exp(-iHΔt)` split into gates", not "what is the
schedule"); `order` is *data*, `trotter_steps` *computes* the `(bond_parity, sub_dt)` substep sequence
(Suzuki recursion handles any even order — no macro, no type-per-order). The abstract `ProductFormula`
leaves room for other splittings (higher-order, commutator-free) without touching the engine.

`ExactDiagonalization` makes the dense/sparse forms (§6) a first-class algorithm rather than a validation
aside, because **Ex 10 is a full ED workflow** (not an MPS exercise): `:ground` calls `eigsolve(sparse(H))`
(KrylovKit Lanczos) for the ground (and low-lying) states (Ex 10c); `:full` calls
`eigen(Hermitian(Matrix(H)))` for the entire spectrum (Ex 10d). ED also carries its own time evolution
(§10): exact propagation of the full state vector, complementary to MPS/TEBD and exact up to the small
system size. It operates on a dense state vector (`statevector(ψ)` — the `local_dim^L` array; `neel_state`
etc. convert trivially), and any operator's many-body matrix — including single-site `Sᶻ_i` and bond
`Sᶻ_iSᶻ_{i+1}` (Ex 10a) — is `Matrix(local_op(…))` / `Matrix(two_point(…))`, the kron-with-identities
embedding the same `Matrix(::Operator)` path already provides. *Deferred:* `TDVP <: Algorithm` — the
variational principle of Ex 11 (quantum geometric tensor, `−g = Sδθ`) realised on the MPS manifold; see §17
and the Ex 11 note in §22.

---

## 10. Time evolution — picture split & evolution protocol

```julia
abstract type Picture end
struct SplitPicture <: Picture; α::Float64; end             # fraction of t carried by the STATE
Schrödinger() = SplitPicture(1.0); Heisenberg() = SplitPicture(0.0); Balanced() = SplitPicture(0.5)

abstract type TimeAxis end
struct RealTime      <: TimeAxis end                         # unitary e^{-iHt}
struct ImaginaryTime <: TimeAxis end                         # non-unitary e^{-Hτ}; renormalise each step

# EvolutionProtocol = the SCHEDULE axis: is H constant in time, or ramping? (orthogonal to TimeAxis)
abstract type EvolutionProtocol{A<:TimeAxis} end
axis(::EvolutionProtocol{A}) where {A} = A()

# constant H, uniform step — the quench / fixed-grid case (all of Ex 7–10)
struct ConstantProtocol{A<:TimeAxis} <: EvolutionProtocol{A}; dt::Float64; nsteps::Int; end
total_time(p::ConstantProtocol) = p.dt * p.nsteps
hamiltonian_at(p::ConstantProtocol, H, ::Int) = H           # same H every step

# FUTURE subtype (deferred, §17): time-dependent ramp, e.g. annealing H(s) = (1−s)·H₀ + s·H₁.
# Carries different data (a ramp s(t) + endpoint Hamiltonians) and rebuilds the gate per step via
# hamiltonian_at — which is why it's a *subtype*, not a flag. The engine loop is unchanged.
# struct AnnealingProtocol{A,F} <: EvolutionProtocol{A}; dt::Float64; nsteps::Int; ramp::F; H₀; H₁; end
# hamiltonian_at(p::AnnealingProtocol, _, k) = (s = p.ramp(k * p.dt / total_time(p)); (1−s)*p.H₀ + s*p.H₁)

# The propagator carries the axis as a type parameter, so the evolution loop
# dispatches on it at compile time and can never apply the wrong step rule.
struct Propagator{A<:TimeAxis, T} <: AbstractOperator        # AbstractOperator: §5 (pending redesign)
    data::T                                                  # the exp'd TensorMap / dense matrix / Krylov action
    dt::Float64
end
_phase(::Type{RealTime})      = -im                          # e^{-i h dt}
_phase(::Type{ImaginaryTime}) = -1                           # e^{-h τ}

# one source of truth: the unitarity property is DERIVED from the axis, never stored twice (cf. §5 traits)
opclass(::Propagator{RealTime})      = Unitary()
opclass(::Propagator{ImaginaryTime}) = PositiveDef()

# gate INHERITS the axis from the protocol, so the gate and the protocol can never disagree
gate(h, dt, ::A) where {A<:TimeAxis} = Propagator{A,typeof(h)}(exp(_phase(A) * dt * h), dt)
gate(h, p::EvolutionProtocol{A}) where {A} = gate(h, p.dt, A())   # one-shot (constant); annealing loops via hamiltonian_at

# the step rule forks on the axis: real time must NOT renormalise, imaginary time MUST
step!(ψ, g::Propagator{RealTime})      = apply!(ψ, g.data)                   # unitary → norm preserved
step!(ψ, g::Propagator{ImaginaryTime}) = (apply!(ψ, g.data); normalize!(ψ))  # contraction → renormalise
```
`Picture` distributes a correlator's evolution between state and operator (Kennes & Karrasch CPC 200, 37
(2016) §2: `⟨A⟩_ψ(t)=⟨ψ(t_ψ)|A(t_A)|ψ(t_ψ)⟩`, `t_ψ+t_A=t`). Balanced `α≈0.5` roughly doubles reachable
time at fixed χ; `α` is tuning data with named endpoints; the Final needs only `Schrödinger()`. `TimeAxis`
is a type because real vs imaginary forks behaviour (unitary/renormalise) and meaning (dynamics vs
ground/thermal projection); the grid is data. The axis is a **type parameter on both `EvolutionProtocol` and the
returned `Propagator`**: `gate(h, p::EvolutionProtocol{A})` reads the axis off the protocol and stamps it onto the
propagator, so the two can never drift apart, and `step!` dispatches on `Propagator{RealTime}` vs
`Propagator{ImaginaryTime}` — structurally guaranteeing real-time never renormalises and imaginary-time
always does. The unitarity property (`Unitary`/`PositiveDef`, §5) is *derived* from the axis via `opclass`,
not stored separately, so there is one source of truth. `h` is a `TensorMap` (block-wise `exp`) in a graded
backend. (`AbstractOperator`/`opclass`/the property traits land with the §5 operator/observable redesign,
still being decided; the axis wiring here is independent of that choice.)

`EvolutionProtocol{A}` itself is *abstract* — it is the **schedule axis**, answering "is `H` constant or
ramping in time?", orthogonal to both `TimeAxis` (real/imaginary) and the `ProductFormula` decomposition
(§9). `ConstantProtocol` (constant `H`, uniform step) covers all of Ex 7–10; a future `AnnealingProtocol`
(§17) is a clean *subtype* — it carries a ramp `s(t)` and endpoint Hamiltonians and rebuilds the gate each
step via `hamiltonian_at(protocol, H, k)`, which `ConstantProtocol` returns unchanged. Subtypes (not a flag)
because the cases differ in stored data and per-step behaviour, whereas `TimeAxis` stays a *parameter*
because it is an orthogonal marker with no extra data — the same trait-vs-subtype split as elsewhere. The
engine loop is identical across protocols; only `hamiltonian_at` differs.

The same `TimeAxis`/`EvolutionProtocol` drive **ED time evolution** (Ex 10e), but with no Trotter splitting or
truncation: the propagator acts on the full state vector exactly, either as a one-shot dense
`exp(∓(i)·dt·Matrix(H))` reused across steps, or — better for larger `L` — matrix-free Krylov
`exponentiate(sparse(H), ∓(i)·dt, ψ)` (KrylovKit) per step. `ImaginaryTime` here is exact projection to the
ground state (renormalise each step), giving an ED counterpart to imaginary-time TEBD. The ED propagator is
the same `Propagator{A}` carrying the axis — only its `data` differs (a dense `exp`'d matrix or a Krylov
action instead of a Trotter gate sequence), so the `step!` renormalisation fork is shared across both
engines. So `EvolutionProtocol` and `TimeAxis` are shared vocabulary; `TEBD` and `ED` are two evolution *engines*
under them.

**Composable time-extension levers** (independent; multiplicativity is prep-9 Q4):
| Lever | Applies to | Gain | Status |
|---|---|---|---|
| Picture split | `Correlator`/quench | ~2× at `α≈0.5` | hook; Schrödinger endpoint = Final |
| Time-translation (two runs → 2t) | equilibrium `Correlator` | ~2× | `equilibrium = true` |
| Backward auxiliary evolution | finite-T purification | slows χ growth | deferred with `ThermalState` |

---

## 11. Tracker — recording during iteration

A tracker is a *callable* the algorithm hands the current state at each step. The algorithm decides *when*;
the tracker decides *what*.

```julia
abstract type AbstractTracker end
struct NoTracker <: AbstractTracker end
(::NoTracker)(ψ, info) = nothing

struct Tracker <: AbstractTracker
    measures::Vector{Pair{Symbol,Function}}; data::Dict{Symbol,Vector}
    steps::Vector; every::Int; _count::Base.RefValue{Int}
end
Tracker(pairs::Pair{Symbol,<:Function}...; every=1) =
    Tracker(collect(pairs), Dict(k=>[] for (k,_) in pairs), [], every, Ref(0))
function (t::Tracker)(ψ, info)
    t._count[] += 1; t._count[] % t.every == 0 || return nothing
    for (name,f) in t.measures
        push!(t.data[name], applicable(f, ψ, info) ? f(ψ, info) : f(ψ))
    end
    push!(t.steps, info.time); return nothing
end

struct TEBDStepInfo;  step::Int; time::Float64; trunc::Float64; end
struct DMRGSweepInfo; sweep::Int; energy::Float64; variance::Float64; end

# Operator observables — any linear ⟨ψ|O|ψ⟩, built via the Operator abstraction (§5,§7)
measure(O::Operator)  = ψ -> expect(ψ, O)        # the uniform bridge: energy, magnetisation, density, ⟨AᵢBⱼ⟩
measure_energy(H)     = measure(H)               # convenience alias (H is an Operator)
measure_sz(site)      = measure(local_op(SpinHalf(), :Sz, site))

# Structural diagnostics — NOT operator expectations (nonlinear in ψ); bespoke functions
measure_entropy(bond) = ψ -> entanglement_entropy(ψ, bond)    # reads bond_svs (see §19)
measure_bonddim()     = ψ -> maximum(bonddims(ψ))
measure_trunc()       = (ψ,info) -> info.trunc
```
`NoTracker` is the null-object default (no hot-loop branch). `Tracker` records named measurements at
frequency `every` into `data`/`steps`. Step-info structs are typed metadata. There are two kinds of measure
and the Tracker accepts both because its measures are arbitrary `ψ -> value` functions: **operator
observables** go through `measure(O::Operator)` → `expect(ψ, O)` (energy, ⟨Sz⟩, density, two-point
functions — all `⟨ψ|O|ψ⟩`, built with the §7 observable constructors); **structural diagnostics**
(entanglement entropy, bond dimension, truncation error) are *not* operator expectations and stay as
bespoke closures. `measure_entropy` reads the stored per-bond `SchmidtSpectrum` (the upgraded `bond_svs`,
now `Vector{SchmidtSpectrum}` rather than a bare vector of singular values) and computes
`entanglement_entropy` on its spectrum; it is free only when the canonical centre is at that bond (else
`O(χ³)` recompute — §19).

---

## 12. The `solve` entry point

```julia
solve(H::Hamiltonian, study::Study, alg::Algorithm,
      protocol::EvolutionProtocol = default_protocol(study),
      tracker::AbstractTracker = NoTracker()) -> Result
struct Result{S}; state::S; observables::Dict{Symbol,Vector}; converged::Bool; end
```
Single entry point; dispatches on `(study, algorithm)`; invalid pairings have no method and error at the
call site. Returns the final state/observable, the tracker `data`, and convergence.

### Validity matrix
| Study | DMRG | TEBD (imag) | TEBD (real) | ED (sparse/dense) | ED (real/imag) | Purification |
|---|---|---|---|---|---|---|
| `GroundState` | ✓ | ✓ | — | ✓ (Ex 10c/d) | ✓ (imag, exact projection) | — |
| `ThermalState` | — | — | — | ✓ (full spectrum) | — | ✓ *(deferred)* |
| `Quench` | — | — | ✓ | — | ✓ (real, Ex 10e) | — |
| `Correlator` | — | — | ✓ (any `picture`) | — | ✓ (small `L`) | — |

`ED` is valid for any study **at small `L`** (cost `local_dim^L`); it is the exact reference the MPS methods
are validated against (§14, §21), and Ex 10's deliverable in its own right.

### Consistency checks `solve` enforces
- **Axis ↔ study:** `EquilibriumStudy` prep by TEBD/ED-imag needs `EvolutionProtocol{ImaginaryTime}`; `DynamicStudy`
  needs `EvolutionProtocol{RealTime}`.
- **Picture ↔ study:** non-Schrödinger `picture` only for `Correlator`.
- **Symmetry ↔ DoF ↔ state:** reject `FermionNumber` on a pairing (U(1)-broken) Hamiltonian; if `H` carries
  a nontrivial symmetry, require a charge-definite `Quench.ψ0` (else warn).
- **ED size guard:** `ED` refuses `local_dim^L` beyond a configurable cap (avoids accidental memory blowup —
  `4^L` for `Electron`; §19/§20).

---

## 15. Design rules (the throughline)

| Decide as a TYPE when… | Decide as DATA when… |
|---|---|
| it forks behaviour (real vs imaginary time; commuting vs anticommuting DoF) | numeric value the math consumes (`dt`, `β`, `α`, `U`) |
| it enforces validity by dispatch (study/algorithm; DoF/symmetry) | content varies but behaviour does not (the measure list) |
| it selects a backend/algebra (DoF, symmetry → `S`) | a tunable swept parameter (`α`, `maxdim`) |

Corollaries: no `@assumptions` macro; no stringly-typed flags (→ types/arrays); "registry" = named
constructors; conversions are `Base` constructors; **DoF is primary, basis change is a separate operation.**

---

## 16. Implementation order (course-driven)

1. `Chain` + `sites`/`bonds`.
2. `Spin{½}` + `operators(SpinHalf())`.
3. `LocalTerm`/`BondTerm`/`Hamiltonian`.
4. `XXZ`/`Ising`/`Heisenberg`.
5. `MPO(H)` — unblocks **Ex 6** and the **Final**.
6. `Matrix(H)`/`sparse(H)` — unblocks **Ex 10** and validation.
7. `SpinlessFermion`/`Majorana` DoFs + `basis_change(_, SpinHalf())` (Jordan–Wigner, Route A);
   `tV`/`kitaev_chain`/`majorana_operators` — unblocks **Ex 3/4**; validate via §14 (delegating to `XXZ`/`Ising`).
8. `EvolutionProtocol`/`TimeAxis`/`gate`, `SuzukiTrotter`/`trotter_steps`, `TEBD`; wire `solve(Quench,TEBD,RealTime)` — the **Final**.
9. `Tracker` + measurement constructors.
10. `solve(GroundState,…)` via DMRG and imaginary-time TEBD; cross-validate.
11. (Later) `Electron`/`hubbard` + native fermionic-graded backend (Route B); `Spin{1}`/AKLT; `Correlator`
    + picture split; `ThermalState` + purification; `U1`/`SU2`/`FermionParity` symmetry; registry
    datastructure; 2D/periodic; `n`-body terms.

---

## 17. Deferred / out of scope (course)

`Electron`/Hubbard + native fermionic backend · `Spin{S>½}` (spin-1/AKLT) · registry datastructure ·
2D/`Torus`/periodic geometry · `ThermalState`/purification + backward-auxiliary disentangler · open/Lindblad
solvers · `Correlator` + picture split + time-translation trick · `U1`/`SU2`/`FermionParity` native
backend · `n`-body (plaquette) terms · general `Majorana`-mode pairing schemes · `d`-level DoF
generalisation · **`AnnealingProtocol <: EvolutionProtocol`** (time-dependent ramp `H(s(t))`; seam =
`hamiltonian_at`, §10) · other **`ProductFormula`** splittings beyond `SuzukiTrotter` (commutator-free,
higher-order; seam = `trotter_steps`, §9) · **density-operator & excitation spectra via `EigValSpectrum`**
(reduced-density-matrix diagonalisation for the full entanglement spectrum, low-lying excitation gaps; the
type and the `AbstractSpectrum` verbs already exist, seam = §13) · **`TDVP <: Algorithm`** (time-dependent variational principle on
the MPS manifold — the tensor-network realisation of Ex 11's variational geometry; see below). Each has a
named seam above.

**Out of scope entirely (different paradigm):** variational Monte Carlo with neural-network quantum states
(Ex 11's hands-on, done in NetKet/Python). This is not a tensor-network method and is not ported into
Qritical.jl. The *conceptual* link is real, though: Ex 11's quantum geometric tensor `Sᵢⱼ` and update
`−g = Sδθ` (stochastic reconfiguration / natural gradient) are the **Dirac–Frenkel variational principle**,
which on the MPS variational manifold *is* `TDVP` — `S` becomes the MPS tangent-space metric and `g` the
energy gradient. So Ex 11 motivates the deferred `TDVP` algorithm above, while the NetKet exercise itself
stays in Python.

---

## 18. Consistency review (full package)

- **Acyclic dependencies;** single cross-link Tracker → step-info. No cycles.
- **One source of truth for operators:** all terms/gates/observables draw from `operators(dof)`.
- **Operator/observable unification:** `Hamiltonian` and observables are one `Operator` type; `expect(ψ,O)`
  is the single contraction for any linear observable. The role (generator vs. measured) is set by use, not
  type. Nonlinear/structural diagnostics (entanglement, bond dim) are deliberately *outside* `Operator` and
  handled as plain `ψ -> value` functions — the Tracker accepts both.
- **DoF is primary;** statistics is intrinsic (`statistics(dof)`); `basis_change` is the sole DoF-to-DoF
  equivalence path. No redundant "representation" field — the DoF type *is* the basis.
- **Symmetry threads via one funnel** (`physical_space(dof, sym)`); downstream code is symmetry- and
  statistics-agnostic.
- **Coherence enforced, not assumed:** `solve` rejects imaginary-time dynamics, real-time ground-state
  prep, `FermionNumber` on pairing models, and symmetry without a charge-definite state.
- **One truncation-vocabulary boundary** (`AbstractTrunc → TruncationScheme`) at the SVD/tsvd site.
- **Form independence:** `MPO`/`Matrix`/`sparse` are interchangeable views; validation checks they agree.
- **Naming uniform:** constructors over `to_*`; named endpoints over magic args; type-vs-data applied
  consistently across `Study`/`Algorithm`/`Picture`/`TimeAxis`/`AbstractDoF`.

---

## 19. Performance bottlenecks & mitigations

1. **SVD dominates the hot loop** (`O(χ³)`/bond/step; χ grows with time — the wall). Levers: truncation,
   symmetry (smaller per-sector blocks), picture split. Not cures.
2. **Term-list type stability** — parametrise `Hamiltonian{…,LT,BT}` on concrete term types.
3. **`Quench.ψ0::Any` / untyped observables** — parametrise `Quench{Ψ}`; cross the dynamic `Dict`/`Vector`
   boundary once via a function barrier.
4. **Heisenberg-picture MPO growth** — MPO–MPO products multiply bond dims; compress every step.
5. **Column-major loop order** — innermost loop over leftmost index (`T[η,σ,λ]`: `λ→σ→η`); `@simd` only after
   the order is right; route big contractions through `mul!`/BLAS (Apple Accelerate); avoid oversized
   intermediates.
6. **Per-step allocation** — preallocate workspace in the `TEBDProblem` iterator state; in-place ops.
7. **Tracker cost** — entropy every step forces an `O(χ³)` recompute unless the centre is at that bond; use
   `every` and align with the sweep direction.
8. **Dense `Matrix(H)`** scales as `local_dim^L` (`2^L` spin, **`4^L` electron** — even smaller `L` reach
   for Hubbard); validation-only; use `sparse(H)` + KrylovKit beyond.
9. **Symmetry overhead at small size** — block bookkeeping can be slower than dense for tiny systems.
10. **Route-A long-range/multi-species strings** widen the MPO; prefer Route B for long-range and for the
    Hubbard model.

---

## 20. Security, robustness & reproducibility

Local scientific library: no network I/O, no credentials → minimal classic attack surface. Real concerns
are correctness, safe input, reproducibility.

- **No `eval`/no injection:** declarative model spec (types/dispatch, no macros/strings) leaves no
  code-injection vector. Never `eval` user strings into operators.
- **Deserialization hazard:** `Serialization.deserialize`/`BSON` execute arbitrary code on load. For saved
  states and fixtures (`docs/src/tutorials/data/`, loaded by `@__DIR__`-relative path), use HDF5/JSON; never
  `deserialize` an untrusted file.
- **Validate parameters; fail loudly:** `L ≥ 1`, `maxdim ≥ 1`, `dt`/`β` finite, array lengths matching
  bond/site counts, `sym` valid for the DoF (reject `FermionNumber` on pairing). Prevents accidental
  resource exhaustion (`Matrix(H)` at large `L`, worse for `Electron`'s `4^L`).
- **Numerical safety:** renormalise imaginary-time steps; finiteness `@assert`s at step boundaries under a
  debug flag; respect parity superselection (`⟨c⟩=0`).
- **Reproducibility:** pin dependency versions, commit `Manifest.toml`; pass an explicit `rng` to
  `disorder_realization`.
- **Path handling:** in the ProcessExerciseSheet automation, validate/normalise derived paths (no traversal).

---

## 21. End-to-end examples

```julia
# Ex 6 — MPO of a site-dependent XXZ chain
g = Chain(20)
H = XXZ(g; J=1.0, Jz=uniform(length(bonds(g)),1.0), h=collect(range(-0.5,0.5; length=g.L)))
W = MPO(H)

# Ex 10 — exact-diagonalisation validation
H8 = XXZ(Chain(8); J=1.0, Jz=1.0, h=0.3)
@assert isapprox(energy(solve(H8, GroundState(), DMRG(64,1e-10,20)).state),
                 eigmin(Matrix(H8)); atol=1e-8)

# Ex 3/4 — basis change (Jordan–Wigner): a fermion model becomes a spin model
Hf = tV(Chain(12); t=1.0, V=1.0, μ=0.5)        # defined on SpinlessFermion
Hs = basis_change(Hf, SpinHalf())              # equivalent spin chain (local NN terms)

# Final — TEBD quench of a Néel state, tracking ⟨Sᶻ⟩ and entropy
g = Chain(20); H = XXZ(g; J=1.0, Jz=1.0)
res = solve(H, Quench(neel_state(g), 5.0),
            TEBD(2, ValCutoffTrunc(1e-10)),
            ConstantProtocol{RealTime}(0.05, 100),
            Tracker(:sz_mid  => measure_sz(g.L÷2),
                    :entropy => measure_entropy(g.L÷2); every=2))
res.observables[:entropy]

# Future — Hubbard model defined on Electron sites (native fermionic backend)
H = hubbard(Chain(16); t=1.0, U=8.0)           # 4D sites; U(1)_c ⊠ U(1)_s ⊠ parity
gs = solve(MPO(H; sym=:u1u1parity), GroundState(), DMRG(400,1e-10,40))
```

---

## 22. Exercise coverage & required additions

Verified against the SS26 sheets Ex1–9 (Schollwöck sections per sheet; Kennes & Karrasch for Ex9) and the
SS24 sheets Ex10 (toy ED) and Ex11 (variational / VMC). Ex1–5 are core MPS primitives; Ex6–9 are this
layer; Ex10 adds the ED engine; Ex11 is analytic + NetKet (out of scope, see below). Everything is covered
**except a handful of Ex6 primitives** below, plus core primitives that should be named so the library is
self-contained, plus the Ex10 ED engine (now in §9–§12).

### Additions needed for Ex6 (this layer)
```julia
# Operator arithmetic — needed for the power-method spectral shift Ĥ − λÎ (Ex6.5)
Base.:+(A::Operator, B::Operator)        # combine term lists (same dof/geom)
Base.:*(λ::Number, A::Operator)          # scale all couplings
identity_operator(g, dof)                # Î as an Operator (one LocalTerm I per site, coupling 1)

# Apply an MPO to an MPS with truncation (Ex6.4); also the power-method workhorse
apply_mpo(O::Operator, ψ; trunc::AbstractTrunc) -> MPS     # ⟨…|MPO(O)|ψ⟩ contracted leg-wise + truncated

# All-to-all observable M = Σᵢ Σⱼ Sᶻᵢ Sᶻⱼ = (Σᵢ Sᶻᵢ)²  (Ex6.2)
magnetization_squared(g; dof=SpinHalf()) -> Operator           # MPO built via the bond-dim-3 finite-state form

# Power method as a GroundState algorithm (Ex6.5)
struct PowerMethod <: Algorithm; shift::Float64; tol::Float64; maxiter::Int; trunc::AbstractTrunc; end
# solve(H, GroundState(), pm::PowerMethod): iterate ψ ← apply_mpo(H - pm.shift*identity_operator(…), ψ; pm.trunc),
#   renormalise, until ‖Δ⟨H⟩‖ < tol; ground-state energy = expect(ψ, H).
```
- `PowerMethod` is the simple iterative ground-state algorithm Ex6 asks for; it reuses `apply_mpo`, `expect`,
  and operator arithmetic, and sits beside `DMRG` under `EquilibriumStudy` in the validity matrix (§12). It
  is the pedagogical precursor to DMRG (Schollwöck §6.1–6.2).
- `magnetization_squared`'s MPO is the standard running-sum (finite-state-machine) construction with bond
  dimension 3 — the §6 claim that "bond dimension encodes the partial-sum structure" must actually deliver
  this for the all-to-all case to be cheap.

### Core primitives to expose (Ex1–5; assumed by this layer)
```julia
ValCutoffTrunc(minval)              # discard singular values σ ≤ minval   (Ex1: 1e-3 / 1e-6)
MaxBondDimTrunc(max_χ)              # keep at most max_χ singular values   (Ex2/Ex3: max bond dim)
overlap(ψ, φ) = dot(ψ, φ)           # ⟨ψ|φ⟩ via canonical contraction     (Ex4.2)
add_mps(a, ψ, b, φ) -> MPS          # a|ψ⟩ + b|φ⟩ via direct sum + truncate (Ex4.4)
is_canonical(ψ; tol) / canonical_error(ψ)   # per-site A/B normalisation check (Ex3.3)
neel_state(g) -> MPS                # product state |↑↓↑↓…⟩               (Ex8/Ex9)
```

### Data I/O & tensor construction (reusable; used from Ex1 onward)
The exercises load serialized tensors (`A.txt`, `psi.jls`, `psi.npy`, `psi1/2.jls`, …). Treat these as
opaque serialized tensor data (like NumPy arrays). A small, reusable I/O layer reads them and **constructs
the code's tensor objects with the correct leg/space structure** — so every exercise that consumes input
data goes through the same path rather than ad-hoc loading.
```julia
load_array(path::AbstractString) -> AbstractArray
# Dispatch on extension: `.jls` → Serialization.deserialize (trusted course data only — see §20);
# `.txt`/`.dat` → DelimitedFiles.readdlm;  `.npy`/`.npz` → NPZ.npzread (Python interop).
# Returns a raw dense array; it does not yet know about MPS/leg conventions.

as_state(data, L::Int; d::Int = 2) -> Array          # reshape a flat dᴸ vector → rank-L tensor (d,…,d)
bipartition_matrix(ψ, cut::Int) -> Matrix            # reshape full state → (d^cut × d^(L−cut)) matrix at a cut (Ex1.3)

to_mps(ψ::AbstractArray, dof = SpinHalf(); trunc::AbstractTrunc,
       form = LeftCanonical) -> FiniteMPS
# Build an MPS from a full state vector/tensor by successive SVD — this IS the Ex2 decomposition, so the
# loader feeds straight into it. Physical legs are kets (`TIx{Upper}`); each site tensor's row/column split
# is the positional Schmidt **bipartition** set by the sweep direction (left-canonical groups (left⊗phys) as
# rows, right as columns), not the operator variance rule — so loaded data inherits the correct state
# convention from the start (§13 state-vs-operator rule).

to_tensormap(data::AbstractArray, codomain, domain) -> TensorMap   # wrap dense data with explicit spaces (§13)

save_results(path, data)             # round-trip YOUR outputs via HDF5/JSON, never Serialization (§20)
```
- `load_array` is the single entry point for input data; `to_mps` is the single bridge from a full state to a
  `FiniteMPS`. Ex1 uses `load_array` + `bipartition_matrix` + `do_svd`; Ex2 uses `load_array` + `to_mps`;
  Ex4 loads two states and calls `overlap`/`add_mps`.
- **Trust boundary (§20):** reading the instructor's `.jls` with `deserialize` is fine because that data is
  trusted; the §20 caution is against deserializing *arbitrary/third-party* tensors. When *you* save results,
  use HDF5/JSON (`save_results`), not `Serialization`, so your own artifacts carry no code-execution risk.
- **Index convention on load (§13):** `to_mps` builds a **state**, so its physical legs are partitioned by
  the SVD/canonical **bipartition** (the sweep direction), not by the variance rule. `to_tensormap` for an
  **operator** assigns its two physical legs by variance (lower/bra/input → domain, upper/ket/output →
  codomain). Loading is one more place this discipline applies, and the one most likely to silently mis-wire
  a freshly built tensor if the state/operator distinction is skipped.

### Exercise → building-block map (reuse chain)
| Ex | Reading | Primary calls | Reuses |
|---|---|---|---|
| 1 | Schollwöck ≤ p18 | `load_array`, `bipartition_matrix`, `do_svd`, `ValCutoffTrunc`, Schmidt rank | — |
| 2 | §4.1.3 | `load_array`+`to_mps`, `CanonicalizeConfig(llim,rlim,trunc)`, `MaxBondDimTrunc` | Ex1 SVD + I/O |
| 3 | §4.4 | `CanonicalizeSweep`, `canonical_error`; 3.4 → `basis_change` (analytic) | Ex1–2 |
| 4 | §4.2,4.2.1,4.3 | `load_array`+`to_mps`, `overlap`, `expect`, `add_mps` | Ex2–3 canonical forms + I/O |
| 5 | §4.5,4.5.1,4.6 | `VidalForm`, `expect`, `two_point` | Ex2–4 |
| 6 | §5,6.1,6.2 | `XXZ`+`MPO`, `magnetization_squared`, `expect`, `apply_mpo`, `PowerMethod` | Ex1–5 |
| 7 | §7,7.1 | `gate`, `trotter_steps`, even/odd update | Ex2–3 (apply gates), Ex1 (truncate) |
| 8 | §7.2 | `Quench`, `TEBD(4,…)`, `ConstantProtocol{RealTime}`, `Tracker(:sz,:entropy)`, `neel_state` | Ex5 (observables/`bond_svs`), Ex7 |
| 9 | Kennes & Karrasch ≤ §3 | `disorder_realization`, `ConstantProtocol{ImaginaryTime}`, `two_point`, `solve(GroundState,TEBD)` | Ex8 wholesale + Ex6 `expect` |
| 10 | ED (own derivation) | `Matrix(local_op/two_point)`, `XXZ`, `solve(_,ED(:ground/:full))`, ED real-time `exp`/`exponentiate` | Ex6 operators, §6 `Matrix`/`sparse` |
| 11 | paper §IID + NetKet | (1,2) analytic; (3) **NetKet/Python**; ED cross-check = `solve(_,ED(:ground))` | Ex10 ED only |

The picture-split/`Correlator` machinery (§8,§10) is required by **none** of Ex1–9 — confirming it is a
correct post-course extension. The final assignment imports the accumulated library; §16's build order
follows this dependency chain exactly.

### Ex 10 & Ex 11 — what they add

**Ex 10 (toy ED)** is the one sheet that promotes the dense/sparse forms from "validation aside" to a
first-class algorithm. Additions, all now in the plan: the `ExactDiagonalization` (`ED`) algorithm with
`:ground` (sparse Lanczos, Ex 10c) and `:full` (dense `eigen`, Ex 10d) modes (§9); ED time evolution under
the shared `EvolutionProtocol`/`TimeAxis` (exact `exp`/Krylov `exponentiate`, Ex 10e, §10); the ED rows in the
validity matrix and the `local_dim^L` size guard (§12). Ex 10a's "operators in the many-body Hilbert space"
need nothing new — `Matrix(local_op(SpinHalf(),:Sz,i))` and `Matrix(two_point(…))` are the kron-embedding,
already provided by `Matrix(::Operator)`. Ex 10b is `XXZ` + `Matrix`. So Ex 10 reuses the operator layer
(Ex 6) and adds only the ED *engine* — and that engine is the exact reference the MPS methods validate
against, so it earns its place library-wide, not just for one sheet.

**Ex 11 (variational / VMC)** adds essentially nothing to the *Julia* package, and it is worth being precise
about why. Parts (1) and (2) are pen-and-paper derivations (the energy gradient
`∂_θE = 2Re[⟨ĤÔᵢ⟩−⟨Ĥ⟩⟨Ôᵢ⟩]` and the quantum geometric tensor / `−g = Sδθ`) — no package code. Part (3) is
done in **NetKet** (Python: an RBM neural-quantum-state ansatz, Metropolis sampling, stochastic
reconfiguration), which is a *different paradigm* from tensor networks and is explicitly "outside the scope"
per the sheet; it is not ported into a TensorKit library. The only piece that touches Qritical.jl is the ED
cross-check (`eigsh` in the notebook ↔ `solve(H, GroundState(), ED(:ground))` here). The deeper, genuinely
useful connection is conceptual and is recorded in §17: the variational geometry of (1)–(2) is the
Dirac–Frenkel principle, which on the MPS manifold *is* TDVP — motivating the deferred `TDVP` algorithm. (As
an aside, Ex 11c's low Metropolis acceptance is because `MetropolisLocal` proposes single-spin flips that
leave the relevant total-`Sᶻ` sector; the fix is `MetropolisExchange`, which swaps neighbours and conserves
magnetisation — a sampler issue, not a Qritical.jl concern.)

---
