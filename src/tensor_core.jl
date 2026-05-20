using TensorOperations
using LinearAlgebra

# ── IndexedTensor ─────────────────────────────────────────────────────────────

# Naming deviates from Julia convention ({T,N}) intentionally:
#   - Element: the scalar field the tensor lives over (e.g. ComplexF64 for
#     wavefunctions, Float64 for real Hamiltonians)
#   - Order: number of indices — distinct from "rank", which we reserve for
#     the linear-algebraic rank used in bond dimension truncation
#   - Element is unconstrained (not <:Number) to stay compatible with AD
#     frameworks (ForwardDiff, Enzyme) that inject non-standard numeric types
"""
    IndexedTensor{Element, Order, D<:AbstractArray{Element,Order}}

An `Order`-index tensor together with a named, directed index for each leg.
The backing store type `D` is a type parameter — `:native` mode uses plain
`Array{Element,Order}`; future backends (e.g. `:tensorkit`) will use other
subtypes of `AbstractArray`.

    IndexedTensor(data::Array{Element,Order}, indices::NTuple{Order, AbstractIndex})
"""
struct IndexedTensor{Element, Order, D<:AbstractArray{Element,Order}} <: AbstractArray{Element,Order}
    data::D
    indices::NTuple{Order, AbstractIndex}
end

Base.size(t::IndexedTensor) = size(t.data)
Base.getindex(t::IndexedTensor, i...) = getindex(t.data, i...)
Base.setindex!(t::IndexedTensor, v, i...) = setindex!(t.data, v, i...)
Base.strides(t::IndexedTensor) = strides(t.data)
function Base.unsafe_convert(::Type{Ptr{Element}}, t::IndexedTensor{Element}) where {Element}
    return Base.unsafe_convert(Ptr{Element}, t.data)
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
    sA.label == sB.label || throw(
        ArgumentError(
            "Label mismatch on index $label: :$(sA.label) vs :$(sB.label)"
        ),
    )
    return local_hilbert_dim(sA) == local_hilbert_dim(sB) || throw(
        DimensionMismatch(
            "Dimension mismatch on index $label: $(local_hilbert_dim(sA)) vs $(local_hilbert_dim(sB))"
        )
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
