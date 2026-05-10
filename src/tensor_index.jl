using TensorOperations
using LinearAlgebra

# ── IndexDirection ────────────────────────────────────────────────────────────

"""
    IndexDirection

Direction of a tensor index, distinguishing contravariant (upper) from covariant
(lower) indices. The following aliases all refer to the same two values:

| Altland & Simons | Dirac  | TensorKit | Visual      | This library    |
|:-----------------|:-------|:----------|:------------|:----------------|
| Contravariant    | Ket    | CoDomain  | `UpIndex`   | `Contravariant` |
| Covariant        | Bra    | Domain    | `DownIndex` | `Covariant`     |

Contractions must always pair one `Contravariant` index with one `Covariant` index.
"""
@enum IndexDirection Contravariant Covariant

const Ket = Contravariant;
const Bra = Covariant;
const UpIndex = Contravariant;
const DownIndex = Covariant;
const CoDomain = Contravariant;
const Domain = Covariant;

"""
    flip(dir::IndexDirection)

Raise or lower the index.
"""
flip(dir::IndexDirection) = dir == Contravariant ? Covariant : Contravariant

# ── AbstractIndex and concrete subtypes ──────────────────────────────────────

"""
    AbstractIndex

Supertype for all tensor index/leg types. Concrete subtypes: `PhysicalIndex`,
`BondIndex`.
"""
abstract type AbstractIndex end

"""
    PhysicalIndex(label, dim, site, dir)

An index representing the local Hilbert space at `site` in a spin chain.
`dim` is the local dimension (2 for spin-1/2). `dir` is `Contravariant` (ket)
or `Covariant` (bra).
"""
struct PhysicalIndex <: AbstractIndex
    label::Symbol
    dim::Int
    site::Int
    dir::IndexDirection
end

"""
    BondIndex(label, dim, dir)

A virtual/bond index carrying entanglement between tensors. `dim` is the bond
dimension D. `dir` is `Contravariant` (outgoing) or `Covariant` (incoming).
"""
struct BondIndex <: AbstractIndex
    label::Symbol
    dim::Int
    dir::IndexDirection
end

is_physical(::PhysicalIndex) = true
is_physical(::BondIndex) = false
is_bond(i::AbstractIndex) = !is_physical(i)

# ── dual, adjoint, directional helpers ───────────────────────────────────────

"""
    dual(i::AbstractIndex)

Return a copy of `i` with the `IndexDirection` flipped. Also available as `i'`
via `Base.adjoint`.
"""
dual(i::PhysicalIndex) = PhysicalIndex(i.label, i.dim, i.site, flip(i.dir))
dual(i::BondIndex) = BondIndex(i.label, i.dim, flip(i.dir))

Base.adjoint(i::AbstractIndex) = dual(i)

"""
    isdual(i, j)

Return `true` if `i` and `j` are the same kind (`PhysicalIndex`/`BondIndex`),
have equal `dim`, and have opposite `dir` — i.e. they form a valid contraction
pair.
"""
isdual(i::T, j::T) where {T<:AbstractIndex} = i.dir != j.dir && i.dim == j.dim
isdual(::AbstractIndex, ::AbstractIndex) = false

"""
    as_covariant(i)    → index with `dir == Covariant`  (idempotent)
    as_contravariant(i) → index with `dir == Contravariant` (idempotent)
"""
as_covariant(i::AbstractIndex) = i.dir == Covariant ? i : dual(i)
as_contravariant(i::AbstractIndex) = i.dir == Contravariant ? i : dual(i)

# ── IndexedTensor ─────────────────────────────────────────────────────────────

"""
    IndexedTensor{T, N}

An N-dimensional array together with a named, directed index for each leg.

    IndexedTensor(data::Array{T,N}, indices::NTuple{N, AbstractIndex})
"""
struct IndexedTensor{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
    indices::NTuple{N,AbstractIndex}
end

Base.size(t::IndexedTensor) = size(t.data)
Base.getindex(t::IndexedTensor, i...) = getindex(t.data, i...)
Base.setindex!(t::IndexedTensor, v, i...) = setindex!(t.data, v, i...)
Base.strides(t::IndexedTensor) = strides(t.data)
function Base.unsafe_convert(::Type{Ptr{T}}, t::IndexedTensor{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, t.data)
end

# ── TensorOperations.jl interface ────────────────────────────────────────────

TensorOperations.tensorstructure(t::IndexedTensor, i::Int, ::Bool) = t.indices[i]

function TensorOperations.checkcontractible(
    sA::AbstractIndex, sB::AbstractIndex, _, _, label
)
    typeof(sA) == typeof(sB) || throw(
        ArgumentError("Cannot contract $(typeof(sA)) with $(typeof(sB)) on index $label"),
    )
    sA.dir != sB.dir || throw(
        ArgumentError(
            "Both legs on index $label are $(sA.dir) — contraction requires a dual pair"
        ),
    )
    return sA.dim == sB.dim || throw(
        DimensionMismatch("Dimension mismatch on index $label: $(sA.dim) vs $(sB.dim)")
    )
end

# ── kronecker_delta ───────────────────────────────────────────────────────────

"""
    kronecker_delta(i::T, j::T) where T <: AbstractIndex

Return an `IndexedTensor` representing the Kronecker delta `δ_ij`. Requires `i`
and `j` to be a dual pair (same kind, same dim, opposite direction).
"""
function kronecker_delta(i::T, j::T) where {T<:AbstractIndex}
    isdual(i, j) || throw(
        ArgumentError(
            "kronecker_delta requires a dual index pair — got $(i.dir) and $(j.dir)"
        ),
    )
    return IndexedTensor(Matrix{Float64}(I, i.dim, i.dim), (i, j))
end

function kronecker_delta(::AbstractIndex, ::AbstractIndex)
    throw(ArgumentError("kronecker_delta requires both indices to be the same kind"))
end
