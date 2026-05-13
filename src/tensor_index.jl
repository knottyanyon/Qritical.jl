# Tensor index machinery for covariant notation.
#
# Design is inspired by SymPy's TensorIndex / TensorIndexType and by the
# hand-calculation conventions in Altland & von Delft (2019). The priority
# here is pedagogical clarity — making the index algebra match the notation
# on paper — rather than raw computational performance.
using TensorOperations
using LinearAlgebra

# ── IndexDirection ────────────────────────────────────────────────────────────

"""
    IndexDirection

Direction of a tensor index in covariant notation, corresponding to the index
position (superscript vs subscript) and its arrow direction in tensor network
diagrams:

| Index position    | Arrow in diagram | Value       |
|:------------------|:-----------------|:------------|
| Superscript (up)  | incoming → •     | `UpIndex`   |
| Subscript (down)  | outgoing • →     | `DownIndex` |

Contractions must always pair one `UpIndex` index with one `DownIndex` index.
"""
@enum IndexDirection UpIndex DownIndex

"""
    flip(dir::IndexDirection)

Raise or lower the index.
"""
flip(dir::IndexDirection) = dir == UpIndex ? DownIndex : UpIndex

# ── AbstractIndex and concrete subtypes ──────────────────────────────────────

"""
    AbstractIndex

Supertype for all tensor index/leg types. Concrete subtypes: `PhysicalIndex`,
`BondIndex`.
"""
abstract type AbstractIndex end

"""
    PhysicalIndex(label, site, dir)

An index representing the local Hilbert space at a lattice `site`.
`dir` is `UpIndex` (superscript) or `DownIndex` (subscript). The local
Hilbert space dimension is computed from `site` via `local_hilbert_dim`.
"""
struct PhysicalIndex <: AbstractIndex
    label::Symbol
    site::AbstractSite
    dir::IndexDirection
end

"""
    BondIndex(label, dim, dir)

A virtual/bond index carrying entanglement between tensors. `dim` is the bond
dimension D. `dir` is `UpIndex` (superscript) or `DownIndex` (subscript).
"""
struct BondIndex <: AbstractIndex
    label::Symbol
    dim::Int
    dir::IndexDirection
end

is_physical(::PhysicalIndex) = true
is_physical(::BondIndex) = false
is_bond(i::AbstractIndex) = !is_physical(i)


"""
    local_hilbert_dim(i::PhysicalIndex)
    local_hilbert_dim(i::BondIndex)

Return the local Hilbert space dimension of index `i`.

For a `PhysicalIndex` the dimension is delegated to the site:
`local_hilbert_dim(i.site)`. For a `BondIndex` it is the stored bond dimension.

```jldoctest
julia> local_hilbert_dim(PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex))   # spin-1/2: 2S+1 = 2
2

julia> local_hilbert_dim(PhysicalIndex(:n, SpinlessBosonicSite(1; n_max_occ=3), DownIndex))  # |0⟩…|3⟩: dim = 4
4

julia> local_hilbert_dim(BondIndex(:α, 16, DownIndex))   # stored directly, no site involved
16
```
"""
local_hilbert_dim(i::PhysicalIndex) = local_hilbert_dim(i.site)
local_hilbert_dim(i::BondIndex) = i.dim  # bond dimension is a free parameter, not derived from a site type


# ── dual, adjoint, directional helpers ───────────────────────────────────────

"""
    dual(i::AbstractIndex)

Return a copy of `i` with the `IndexDirection` flipped. Also available as `i'`
via `Base.adjoint`.
"""
dual(i::PhysicalIndex) = PhysicalIndex(i.label, i.site, flip(i.dir))
dual(i::BondIndex) = BondIndex(i.label, i.dim, flip(i.dir))

Base.adjoint(i::AbstractIndex) = dual(i)

"""
    isdual(i, j)

Return `true` if `i` and `j` are the same kind (`PhysicalIndex`/`BondIndex`),
have equal `dim`, and have opposite `dir` — i.e. they form a valid contraction
pair.
"""
isdual(i::T, j::T) where {T<:AbstractIndex} = i.dir != j.dir && local_hilbert_dim(i) == local_hilbert_dim(j)
isdual(::AbstractIndex, ::AbstractIndex) = false

"""
    as_down(i) → index with `dir == DownIndex` (idempotent)
    as_up(i)   → index with `dir == UpIndex`   (idempotent)
"""
as_down(i::AbstractIndex) = i.dir == DownIndex ? i : dual(i)
as_up(i::AbstractIndex) = i.dir == UpIndex ? i : dual(i)

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
    return local_hilbert_dim(sA) == local_hilbert_dim(sB) || throw(
        DimensionMismatch("Dimension mismatch on index $label: $(local_hilbert_dim(sA)) vs $(local_hilbert_dim(sB))")
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
    d = local_hilbert_dim(i)
    return IndexedTensor(Matrix{Float64}(I, d, d), (i, j))
end

function kronecker_delta(::AbstractIndex, ::AbstractIndex)
    throw(ArgumentError("kronecker_delta requires both indices to be the same kind"))
end

# ── Bisection ─────────────────────────────────────────────────────────────────

"""
    Bisection(left, right)
    Bisection(left, ndims)
    Bisection(tensor::IndexedTensor, left_indices)
    Bisection(tensor::IndexedTensor, T::Type{<:AbstractIndex})

A partition of a tensor's indices into two disjoint sets `left` and `right`.

The integer forms take explicit dimension positions; `ndims` infers `right` as
the complement of `left` within `1:ndims`. The `IndexedTensor` forms resolve
positions automatically: pass a vector of `AbstractIndex` objects to specify
the left set by identity, or pass an index type (`PhysicalIndex`/`BondIndex`)
to put all legs of that kind on the left.

# Examples
```jldoctest
julia> b = Bisection([1, 2], 5)
Bisection([1, 2], [3, 4, 5])

julia> b.left
2-element Vector{Int64}:
 1
 2

julia> b.right
3-element Vector{Int64}:
 3
 4
 5

julia> Bisection([1], [1, 2])
ERROR: ArgumentError: Indices [1] appear in both partitions.
[...]
```
"""
struct Bisection
    left::Vector{Int}
    right::Vector{Int}

    function Bisection(left::Vector{Int}, right::Vector{Int})
        shared = intersect(left, right)
        isempty(shared) ||
            throw(ArgumentError("Indices $shared appear in both partitions."))
        length(unique(left)) == length(left) ||
            throw(ArgumentError("left indices contain duplicates."))
        length(unique(right)) == length(right) ||
            throw(ArgumentError("right indices contain duplicates."))
        all(>(0), left) || throw(ArgumentError("left indices must be positive integers."))
        all(>(0), right) || throw(ArgumentError("right indices must be positive integers."))
        return new(left, right)
    end
end

function Bisection(left::AbstractVector{Int}, right::AbstractVector{Int})
    return Bisection(collect(left), collect(right))
end
function Bisection(left::AbstractVector{Int}, ndims::Int)
    return Bisection(collect(left), setdiff(1:ndims, left))
end

function Bisection(tensor::IndexedTensor, left::AbstractVector{<:AbstractIndex})
    left_positions = map(left) do idx
        pos = findfirst(==(idx), tensor.indices)
        isnothing(pos) && throw(ArgumentError("Index $idx not found in tensor"))
        pos
    end
    return Bisection(left_positions, ndims(tensor))
end

function Bisection(tensor::IndexedTensor, ::Type{T}) where {T<:AbstractIndex}
    left_positions = findall(i -> i isa T, collect(tensor.indices))
    isempty(left_positions) && throw(ArgumentError("No indices of type $T found in tensor"))
    return Bisection(left_positions, ndims(tensor))
end
