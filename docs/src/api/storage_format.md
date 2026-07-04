# Storage Formats

A [`LatticeOperator`](@ref) is stored *lazily* as a list of `OneSiteTerm`s and
`TwoSiteTerm`s over a geometry and DoF. Turning that term list into an actual matrix —
via [`matrix_repr`](@ref) — can be done two ways, and the right choice depends entirely
on what happens to the matrix afterwards:

- A **dense** ``d^L \times d^L`` array is convenient for full diagonalisation
  (`LinearAlgebra.eigen`) and for small systems where you want to inspect every entry.
- A **sparse** `SparseMatrixCSC` avoids ever materialising the ``d^{2L}`` zero entries
  of a nearest-neighbour Hamiltonian, which is essential once `KrylovKit.eigsolve` (or
  any other Lanczos/Krylov solver) is doing the work — those solvers only need
  matrix-vector products, never the full matrix.

Rather than have `matrix_repr` guess which one you want, the format is passed
explicitly as a **tag type** — an instance of [`StorageFormat`](@ref) — so the choice
is visible at the call site and dispatched on at compile time:

```julia
Hd = matrix_repr(H)                # defaults to DenseFormat()
Hs = matrix_repr(H, SparseFormat())
```

This mirrors the same "tag dispatch, not a `Bool` or `Symbol` flag" pattern used
elsewhere in Qritical (e.g. [`TimeAxis`](@ref), [`CanonicalRelation`](@ref)): adding a
future backend — a `BlockSparseFormat` for TensorKit graded spaces carrying quantum
number sectors, say — is a new subtype, not a change to every call site that already
uses `matrix_repr`.

---

## Types

```@docs
StorageFormat
DenseFormat
SparseFormat
```
