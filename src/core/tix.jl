#=META
source:
  author: Bavithra
  coauthor: N/A
  reviewer:
docstrings:
  author: Bavithra
  coauthor:
  reviewer:
refs:
credits: [ITensor.jl](https://github.com/ITensor/ITensors.jl)
=#

"""
    AbstractIx

Root of the Qritical.jl index hierarchy.

Every concrete subtype must implement two methods that expose the index's identity within a
tensor:

  - [`dim(::AbstractIx)`](@ref)   - positive integer size of the index space. Required to
    check index compatibility during contraction and to reshape tensors for matrix operations
    (SVD, matrix-vector products).
  - [`label(::AbstractIx)`](@ref) - symbolic name used for contraction matching. Two indices
    contract only if they have identical labels; the label is the key lookup in the
    contraction machinery.

Concrete subtypes: [`TIx`](@ref), [`MulTIx`](@ref).
"""
abstract type AbstractIx end   # root of the index hierarchy

# abstract interface stubs
dim(i::AbstractIx) = error("`dim` not implemented for $(typeof(i))")
label(i::AbstractIx) = error("`label` not implemented for $(typeof(i))")

"""
    TIx <: AbstractIx

A single tensor leg carrying a symbolic `label` and a positive `dim`ension.

`TIx` no longer carries any variance (domain/codomain) tag; index placement in a `TensorMap`'s
codomain or domain is decided structurally at construction time (see `QTensor`), not encoded
in the index type itself. Two `TIx` values are equal exactly when their `label` and `dim`
match.

# Fields

  - `label :: Symbol` - name used for contraction matching (e.g. `:α`, `:σ_1`)
  - `dim   :: Int`    - strictly positive size of the index space (`dim ≥ 1`)

# Examples

```jldoctest
julia> α = TIx(:α, 4);

julia> dim(α)
4

julia> label(α)
:α

julia> α == TIx(:α, 4)
true

julia> α == TIx(:β, 4)
false
```
"""
struct TIx <: AbstractIx # abstract penrose index 
    label::Symbol
    dim::Int        # local Hilbert space dimension; must be > 0 (checked in the constructor)
    function TIx(label::Symbol, dim::Int)
        #TODO - would adding another implementation for dim with symbolic d give some flexibility in contraction cost estimation?
        dim > 0 || throw(ArgumentError("TIx dim must be a positive integer, got $dim."))
        return new(label, dim)
    end
end

"""
    dim(ix::TIx) -> Int

Return the dimension of index `ix`.

# Examples

```jldoctest
julia> dim(TIx(:α, 4))
4

julia> dim(TIx(:σ, 2))
2
```
"""
dim(ix::TIx) = ix.dim

"""
    label(ix::TIx) -> Symbol

Return the symbolic label of index `ix`.

# Examples

```jldoctest
julia> label(TIx(:α, 4))
:α

julia> label(TIx(:σ_1, 2))
:σ_1
```
"""
label(ix::TIx) = ix.label

# Extend Base.== and Base.hash so indices can be stored in dicts/sets and compared.
# Two indices are equal iff they share the same label and dimension. The hash enables fast
# index lookup when building contraction networks; dim is included so different dimensions
# don't collide, even if they happen to have the same label.
Base.:(==)(a::TIx, b::TIx) = a.label == b.label && a.dim == b.dim
Base.hash(ix::TIx, h::UInt) = hash(ix.label, hash(ix.dim, h))

# SECTION -  helper functions for batch construction

"""
    ixs(pairs::Pair{Symbol,Int}...) -> Tuple{TIx,...}

Construct a tuple of indices from `label => dim` pairs.

Calling with no arguments returns an empty tuple `()`, which is useful when chaining with
`filter` or `map` that may produce no results.

## Implementation

Uses Julia's broadcasting operator `.` to apply the constructor element-wise: `first.(pairs)`
extracts labels, `last.(pairs)` extracts dimensions, and `TIx.(...)` broadcasts the
constructor to create each index.

# Examples

```jldoctest
julia> ixs()
()

julia> α, β = ixs(:α => 2, :β => 4);

julia> dim(α), dim(β)
(2, 4)

julia> label(β)
:β
```
"""
ixs(pairs::Pair{Symbol,Int}...) = TIx.(first.(pairs), last.(pairs))
"""
    ixs_range(base::Symbol, dim::Int, last::Int, start::Int=1) -> Tuple{TIx,...}

Construct a tuple of indices with auto-generated labels `base_start`, `base_(start+1)`, ...,
`base_last`, all sharing the same `dim`.

Useful for building a sequence of bond or physical indices along a chain of `L` sites, e.g.
`ixs_range(:χ, D, L)` for `L` virtual bond legs of bond dimension `D`.

# Arguments

  - `base`  - base symbol; the `i`-th label is `Symbol(base, :_, i)`
  - `dim`   - dimension shared by every index in the tuple
  - `last`  - upper bound of the range (inclusive)
  - `start` - lower bound of the range (default: `1`)

# Examples

```jldoctest
julia> r = ixs_range(:χ, 4, 3);

julia> length(r)
3

julia> label(r[1]), label(r[3])
(:χ_1, :χ_3)

julia> all(dim.(r) .== 4)
true

julia> r2 = ixs_range(:α, 2, 5, 3);

julia> length(r2)
3

julia> label(r2[1]), label(r2[3])
(:α_3, :α_5)
```
"""
function ixs_range(base::Symbol, dim::Int, last::Int, start::Int=1)
    return Tuple(TIx(Symbol(base, :_, i), dim) for i in start:last)   # (immutable, compile-time-sized)
end
