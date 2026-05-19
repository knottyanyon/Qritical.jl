# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
julia --project test/runtests.jl

# Run a single test file
julia --project -e 'using Qritical, Test; include("test/test_tensor_index.jl")'

# Format code (Blue style)
julia --project -e 'using JuliaFormatter; format(".")'

# Build docs locally (no live reload)
julia --project=docs docs/make.jl

# Build and serve docs with live reload
julia --project=docs -e 'using LiveServer; servedocs()'
```

## Architecture

Qritical.jl is a Julia package for quantum many-body tensor network computations. The physics focus is spin chains, entanglement, and tensor decompositions.

### Source layout

- **`src/tensor_index.jl`** — Core index and tensor types. The main data model. Contains `IndexDirection`, `AbstractIndex`, `PhysicalIndex`, `BondIndex`, `IndexedTensor`, and `Bisection`. All other modules depend on this.
- **`src/schmidt_decomposition.jl`** — SVD-based decompositions: `factorize_with_svd`, `reshape_tensor_for_bipartition`, `get_schmidt_coefficients`, `get_entanglement_entropy`. Works with both raw arrays and `Bisection` objects.
- **`src/QriticalUtils/`** — Submodule. `read_write_utils.jl` handles `.jls` serialization (`load_state_from_file`, `save_jls`). These load raw `Array{Float64, N}` with no index metadata.
- **`src/hinton_recipe.jl`** — Makie plot recipes for Hinton diagrams.
- **`src/contraction_benchmaking.jl`** — Benchmarking helpers for tensor contractions.

### Key design concepts

**Index system:** Indices carry direction (`Contravariant`/`Covariant`, aliased as `Ket`/`Bra`, `UpIndex`/`DownIndex`, `CoDomain`/`Domain`). Contractions are only valid between a dual pair — same type (`PhysicalIndex`/`BondIndex`), same dimension, opposite direction. Validated via `TensorOperations.checkcontractible`.

**`PhysicalIndex` vs `BondIndex`:** `PhysicalIndex` represents the local Hilbert space at a lattice site (currently stores `site::Int`; being upgraded to `site::AbstractSite`). `BondIndex` represents virtual entanglement legs between tensors.

**`IndexedTensor`:** A thin wrapper around `Array{T,N}` carrying an `NTuple{N, AbstractIndex}`. Implements `AbstractArray` and the `TensorOperations.jl` interface, enabling `@tensor` contraction syntax with index-direction validation.

**`Bisection`:** Partitions a tensor's leg positions into two disjoint sets (left/right). Used to reshape N-dimensional tensors into matrices for SVD. Can be constructed from position integers, `AbstractIndex` objects, or an index type.

**Site types (in progress on branch `claude/build-indices-structs-Lf2V5`):** `AbstractSite` hierarchy (`SpinSite`, `SpinlessFermionicSite`, `HardCoreBosonicSite`, `BosonicSite`) wraps TensorKit `GradedSpace` to encode symmetry sector structure. `PhysicalIndex` is being upgraded to reference an `AbstractSite` instead of a bare `Int`, enabling TensorKit-backed contraction validation and block-diagonal decompositions.

### Dependencies

- **TensorOperations.jl** — `@tensor` macro for contraction syntax; `IndexedTensor` implements its interface.
- **TensorKit.jl** — Provides `GradedSpace`, `U1Space`, `SU2Space`, `Z2Space` for symmetry-graded Hilbert spaces. Used by site types.
- **HalfIntegers.jl** — `HalfInt` type for half-integer spin values. Transitive dependency via TensorKit; add explicitly if used directly.

### Testing

Tests use `Aqua.jl` for code quality checks (ambiguities, stale deps, etc.) in addition to standard `@testset` blocks. New features follow TDD: failing tests are written first in `test/test_tensor_index.jl`, then implemented. The branch for current active development is `claude/build-indices-structs-Lf2V5`.
