# Degrees of Freedom

A **[`degree of freedom`](../references/glossary.md#degree-of-freedom)** ([`DoF`](../references/glossary.md#dof)) is what the model places at each lattice site: the local
Hilbert space, its operator algebra, and — crucially — the **intrinsic inter-site
statistics**. This is *not* a representation choice; it is the physical content of the
model. A spin-½, a spin-1, a spinless fermion, a Hubbard electron, and a hard-core boson
are genuinely different local Hilbert spaces; they are not the same object "viewed
differently".

The distinction that does the most work is the [`canonical relation`](../references/glossary.md#canonical-relation) trait:

- **[`CCR`](../references/glossary.md#ccr)** (spins, hard-core bosons): operators on different sites commute,
  ``[O_i, O_j] = 0`` for ``i \neq j``. No string correction is needed when assembling
  multi-site terms.

- **[`CAR`](../references/glossary.md#car)** (fermions, Majoranas): operators on different sites anticommute,
  ``\{O_i, O_j\} = 0`` for ``i \neq j``. This is handled by one of two routes:

  - **Route A — Jordan–Wigner (dense):** `basis_change(H, SpinHalf())` rewrites a
    fermionic Hamiltonian as a spin model by inserting ``\sigma^z`` strings. The
    result lives in the `Spin{1/2}` DoF and is processed by the standard MPS machinery.
    This is the course route (Ex 3/4).

  - **Route B — native fermionic grading (Week 12):** upgrade the bond legs to
    parity-graded `ElementarySpace`s; the TensorKit backend supplies every ``-1`` sign
    automatically via categorical braiding.

All operator matrices returned by `operators(dof)` are on-site matrices in the natural
basis. They do *not* encode inter-site anticommutation — that is supplied by Route A or B
as needed.

### Sign convention for `Electron`

The local basis is ordered ``\{|0\rangle, |\!\uparrow\rangle, |\!\downarrow\rangle,
|\!\uparrow\downarrow\rangle\}`` with the doubly-occupied state defined as
``|\!\uparrow\downarrow\rangle \equiv c^\dagger_\uparrow c^\dagger_\downarrow |0\rangle``
(spin-up created first). This is the convention used by ITensor's `Electron` site and
Essler et al. It fixes the intra-site sign:

```math
c_\downarrow |\!\uparrow\downarrow\rangle = -|\!\uparrow\rangle
```

---

## Quick Reference

**DoF types:** [`AbstractDoF`](@ref) · [`Spin`](@ref) · [`SpinHalf`](@ref) · [`SpinOne`](@ref) · [`SpinlessFermion`](@ref) · [`Electron`](@ref) · [`MajoranaFermion`](@ref) · [`HardCoreBoson`](@ref)

**Statistics:** [`CanonicalRelation`](@ref) · [`CCR`](@ref) · [`CAR`](@ref)

**Functions:** [`local_dim`](@ref) · [`canonical_relation`](@ref) · [`algebra_generators`](@ref) · [`physical_space`](@ref) · [`NoSymmetry`](@ref)

---

## Types

### Abstract interface

```@docs
AbstractDoF
```

### Concrete degrees of freedom

```@docs
Spin
SpinHalf
SpinOne
SpinlessFermion
Electron
MajoranaFermion
HardCoreBoson
```

### Statistics traits

```@docs
CanonicalRelation
CCR
CAR
```

### Symmetry

```@docs
NoSymmetry
```

## Functions

```@docs
local_dim
canonical_relation
algebra_generators
physical_space
```
