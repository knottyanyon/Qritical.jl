# ADR 0007: MPS Type

## Status

Accepted (implementation in PLAN 7)

## Why this exists

The course exercises require working with full matrix product states, not just
individual tensors at a single bipartition. Canonical forms (Exercise 2), TEBD
(Exercise 3), and observables (Exercise 4) all operate on a *chain* of tensors
with specific index connectivity and normalisation structure. Without an MPS
type, all of this bookkeeping falls to the caller, which is both error-prone
and opaque.

The building blocks have been designed first (Isometry in PLAN 3,
TruncationResult in PLAN 5, SchmidtSpectrum in PLAN 6) precisely so that the
MPS type can use them as primitives rather than re-implementing them.

## Why not dedicated Ket and Bra types

A Ket is an `IndexedTensor` with all physical indices as `DownIndex`. A Bra is
the same with `UpIndex`. The existing `isdual` check already prevents
contracting two kets (they'd have the same direction), and `dual(t)` / `adjoint`
already flips all index directions. A `Ket` wrapper type would be purely
semantic — it would add a layer without adding any dispatch power or invariant
that the index direction system does not already enforce. The risk of premature
abstraction outweighs the readability gain, especially before the MPS type
exists to make the context concrete. Deferred indefinitely.

## Storage: Vector{IndexedTensor}, not Vector{Isometry}

`Isometry` (PLAN 3) is a claim about the *output* of an SVD — a semantic
annotation applied at the point of production. It is not a long-term storage
format. The MPS struct stores `Vector{IndexedTensor}` for three reasons:

1. **In-place operations**: TEBD and canonicalization overwrite individual site
   tensors. Re-wrapping as `Isometry` after every update adds overhead and
   complicates the mutation model.
2. **General-form MPS**: An MPS just constructed from a product state or read
   from disk has no canonical structure — there is nothing to claim isometry
   about until a canonicalization sweep is run.
3. **Consistency with ITensors.jl**: ITensors stores plain tensors in the MPS
   and tracks canonical form separately. The Isometry wrapper is used only when
   returning the U and Vt factors from individual SVD calls (per ADR 0004).

## Canonical form tracking

Canonical form status is tracked via a single field:

```julia
orthogonality_center::Union{Int, Nothing}
```

- `nothing` — general form, no structure assumed
- `0` — right-canonical (all tensors right-isometric; OC conceptually left of site 1)
- `i ∈ 1:L` — mixed canonical with OC at site i
- `L+1` — left-canonical (all tensors left-isometric; OC conceptually right of site L)

This is a *claim*, not a guarantee — floating point drift means a tensor that
was isometric when produced may no longer be after many operations. The same
philosophy as PLAN 3's `Isometry`: the field records what the algorithm
asserted, not what is numerically verified. Use `check_isometry` on individual
tensors when verification is needed.

## Open boundary conditions only

The boundary tensors (sites 1 and L) have one physical index and one bond
index each; bulk tensors have one physical and two bond indices. Periodic
boundary conditions (trace instead of open ends) require a different
contraction structure and are out of scope for the course exercises.

## Mutable struct

Canonical form sweeps and TEBD overwrite individual site tensors. A mutable
struct allows in-place modification without reallocating the full vector. The
cost is that MPS objects cannot be freely shared (mutation is visible to all
holders), but this matches the expected usage pattern in the course exercises.

## Index structure at each site

Following the existing covariant convention (`UpIndex` = incoming, `DownIndex` = outgoing):

- **Left boundary (site 1)**: `[σ₁↓, α₁↓]` — one physical (outgoing), one
  right bond (outgoing to site 2)
- **Bulk (site i)**: `[αᵢ₋₁↑, σᵢ↓, αᵢ↓]` — left bond (incoming from site
  i−1), physical (outgoing), right bond (outgoing to site i+1)
- **Right boundary (site L)**: `[αₗ₋₁↑, σₗ↓]` — left bond (incoming), physical

The `BondIndex` `from`/`to` fields encode directionality: the bond between
sites i and i+1 has `from=i, to=i+1`. The left tensor at site i sees it as a
`DownIndex` (outgoing); the right tensor at site i+1 sees it as `UpIndex`
(incoming). This is consistent with the existing `BondIndex` arrow semantics
from ADR 0002.

## Operations in scope for PLAN 7

- Construction from a list of `IndexedTensor` site tensors
- Basic accessors: `Base.length`, `bond_dim(mps, bond)`, `local_hilbert_dim(mps, site)`
- `LinearAlgebra.norm(mps)` — full contraction ⟨ψ|ψ⟩^{1/2}
- `left_canonicalize!(mps; spec)` — left-to-right SVD sweep, returns `TruncationLog`
- `right_canonicalize!(mps; spec)` — right-to-left sweep, returns `TruncationLog`
- `canonicalize!(mps, center::Int; spec)` — mixed canonical form at site `center`
- `schmidt_spectrum(mps, bond::Int)` — `SchmidtSpectrum` at bond i (requires
  mixed canonical form; asserts `orthogonality_center == bond` or adjacent)
- `entanglement_entropy(mps, bond::Int)` — derived from `schmidt_spectrum`

TEBD, expectation values, and two-site operations are deferred to later plans.

## What is deferred

- **Periodic boundary conditions** — different contraction structure
- **TEBD / time evolution** — PLAN 8 or later
- **Expectation values / observables** — Exercise 4, later plan
- **Infinite MPS (iMPS)** — not covered in the course
- **Two-site update / DMRG** — not covered in the course scope
- **Symmetry-aware MPS** — U(1), SU(2); explicitly out of scope per CLAUDE.md

## References

- Schöllwöck, Ann. Phys. 326 (2011), §4.4 — MPS canonical forms, sweeps
- Von Delft/LMU course notes, MPS-II.3 — iterative diagonalization, MPS structure
- ADR 0002 (`0002-bond-index-arrow-orientation.md`) — BondIndex from/to convention
- ADR 0003 (`0003-backend-dispatch-scoped-context.md`) — backend context
- ADR 0004 (`0004-special-tensor-types.md`) — Isometry as SVD output, not storage
- ADR 0005 (`0005-svd-truncation-tracking.md`) — TruncationResult/Log used in sweeps
- ADR 0006 (`0006-schmidt-spectrum-type.md`) — SchmidtSpectrum used at bonds
- GitHub issue #65
