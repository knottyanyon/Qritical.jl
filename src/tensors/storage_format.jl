# Storage-format tags — control how a LatticeOperator is materialised into a matrix.
#
# Keeping these tags in a dedicated file makes it easy to add new storage backends
# (e.g. BlockSparseFormat for TensorKit graded spaces) without touching the operator
# or ED layers.

"""
    StorageFormat

Abstract supertype for matrix storage-format tags.

A `StorageFormat` tag is passed as the second argument to [`matrix_repr`](@ref) to
select how a [`LatticeOperator`](@ref) is materialised into a concrete matrix:

  - [`DenseFormat`](@ref)  — standard Julia `Matrix{ComplexF64}` (column-major dense array).
  - [`SparseFormat`](@ref) — `SparseMatrixCSC{ComplexF64}` from `SparseArrays`.

Future backends (e.g. `BlockSparseFormat` for TensorKit graded spaces carrying
quantum-number sectors) will be added as new subtypes without changing existing code.

See also: [`matrix_repr`](@ref), [`DenseFormat`](@ref), [`SparseFormat`](@ref)
"""
abstract type StorageFormat end   # abstract type = Python ABC; no instances; exists only as a dispatch label so functions can accept any storage format; adding a new format only requires a new subtype + new methods

"""
    DenseFormat <: StorageFormat

Storage-format tag requesting a dense `Matrix{ComplexF64}`.

Pass to [`matrix_repr`](@ref) to build the full ``d^L \\times d^L`` dense Hamiltonian
matrix via Kronecker products.  This is the default when no format is specified.

Practical limit: ``d^L \\lesssim 2^{20}`` (``L \\approx 20`` for ``d = 2``); larger
systems exhaust memory.

See also: [`SparseFormat`](@ref), [`matrix_repr`](@ref)
"""
struct DenseFormat <: StorageFormat end   # singleton struct: zero-size, no fields; `<: StorageFormat` makes it a subtype; the VALUE `DenseFormat()` is the tag you pass to select dense storage; Python: `class DenseFormat(StorageFormat): pass`

"""
    SparseFormat <: StorageFormat

Storage-format tag requesting a sparse `SparseMatrixCSC{ComplexF64}`.

Pass to [`matrix_repr`](@ref) to assemble the Hamiltonian matrix using sparse
Kronecker products.  Suitable for Lanczos/Krylov solvers (`KrylovKit.eigsolve`)
that only need matrix-vector products and never form the full dense matrix.

See also: [`DenseFormat`](@ref), [`matrix_repr`](@ref)
"""
struct SparseFormat <: StorageFormat end   # singleton struct for the sparse format tag; `SparseMatrixCSC` = Compressed Sparse Column format (standard format for Julia sparse arrays, analogous to `scipy.sparse.csc_matrix`)
