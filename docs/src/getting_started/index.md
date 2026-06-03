# Getting Started

Qritical.jl is a Julia package for matrix product state (MPS) algorithms, built around a small set of composable abstractions:

- **Named, directed indices** — every tensor leg carries a label and a covariant/contravariant position, closing the gap between the diagram and the code.
- **`IndexedTensor`** — pairs a plain Julia array with a tuple of `AbstractIndex` values, one per dimension.
- **`tensor_svd` with truncation** — decomposes any `IndexedTensor` by a chosen bipartition of its legs, applies a truncation strategy, and returns the exact truncation error.

## Sections

| Page | What you will learn |
|:-----|:--------------------|
| [Installation](@ref) | How to add Qritical to a Julia environment |
| [Indices & IndexedTensor](@ref) | `TIx`, `MultiIx`, `IndexedTensor`, `Partition`, `Bipartition`, `group_legs` |
| [SVD & Truncation](@ref) | `tensor_svd`, `KeepFirst`, `KeepAbove`, `KeepRelative`, `BondIndex` |

The [Home](@ref) page has a full auto-generated API reference for every exported symbol.
