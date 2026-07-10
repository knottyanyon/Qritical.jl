# Getting Started

Qritical.jl is a Julia package for matrix product state (MPS) algorithms, built around a small set of composable abstractions:

- **Named, directed indices** — every tensor leg carries a label and a covariant/contravariant position, closing the gap between the diagram and the code.
- **`QTensor`** — pairs a plain Julia array with a tuple of `AbstractIx` values, one per dimension.
- **`do_svd` with truncation** — decomposes any `QTensor` by a chosen bipartition of its legs, applies a truncation strategy, and returns the exact truncation error.

## Pages in this section

| Page | What you will learn |
|:-----|:--------------------|
| [Installation](@ref Installation) | How to add Qritical to a Julia environment |
| [GS-1: Tensors & Indices](gs1_tensors_and_indices.md) | `TIx`, `MulTIx`, `QTensor`, `Partition`, `Bipartition`, `group_legs` |
| [GS-2: SVD & Truncation](gs2_svd_and_truncation.md) | `do_svd`, `NoTrunc`, `MaxBondDimTrunc`, `ValCutoffTrunc` |
| [GS-3: Drawing Tensor Networks](gs3_drawing_tensor_networks.md) | `schematic`, `tensor!`, `bond!`, `leg!`, `partition!`, `note!` |
