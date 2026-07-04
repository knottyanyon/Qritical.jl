# QTensor

A raw multi-dimensional array has no memory of what its axes mean. In tensor-network computations this quickly becomes dangerous: it is easy to contract the wrong pair of legs, to forget which index is the physical spin and which is the virtual [`bond`](../references/glossary.md#bond), or to apply a gate to the wrong site. `QTensor` solves this by bundling a numerical array with an ordered tuple of typed index legs (`TIx{Upper}` or `TIx{Lower}`), making every contraction label-driven rather than position-driven.

The `{D}` parameter on `QTensor{T,N,D}` selects the execution backend without changing any algorithm code:

- **`:native` backend** — `D = Array{T,N}`, the default dense path. Full-rank tensors, all sectors treated as one block. Used for the learning path, ED validation, and any model without exploitable symmetry.
- **`:tensorkit` backend** (Part 2) — `D = TensorKit.TensorMap`. The same `QTensor` wrapper, but the underlying array is [`block-sparse`](../references/glossary.md#block-sparse): each block corresponds to a symmetry sector (a good quantum number), and contractions run *within* blocks. This is how U(1) charge conservation or SU(2) spin symmetry translates into a computational speedup at large [`bond dimension`](../references/glossary.md#bond-dimension) ``\chi``.

The switch between backends is a single `ScopedValue` context (`with_backend(:tensorkit) do … end`) — no algorithm code changes.

---

## Quick Reference

**Types:** [`QTensor`](@ref) · [`Partition`](@ref) · [`Bipartition`](@ref)

**Functions:** [`complement`](@ref) · [`bipartition`](@ref) · [`group_legs`](@ref) · [`dagger`](@ref)

---

## Types

### Main tensor type

```@docs
QTensor
```

### Adjoint and conjugation

```@docs
dagger
Base.adjoint(::QTensor)
```

### Partitions and reshaping

A `Partition` records an ordered group of legs (an axis list for a matricisation), and `Bipartition` pairs a row-partition with a column-partition for use in SVD. `group_legs` permutes and reshapes a `QTensor` to produce a rank-2 tensor whose two `MulTIx` legs encode the full original structure, ready for factorisation.

```@docs
Partition
Bipartition
complement
bipartition
group_legs
```
