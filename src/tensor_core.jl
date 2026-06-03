using LinearAlgebra
using TensorOperations

# Naming deviates from Julia convention ({T,N}) intentionally:
#   Element — the scalar field (e.g. ComplexF64), not constrained to <:Number
#             so AD frameworks (ForwardDiff, Enzyme) can inject dual numbers.
#   Order   — number of indices; we reserve "rank" for the SVD / linear-algebra
#             sense (number of non-zero singular values).

"""
    IndexedTensor{Element, Order, D<:AbstractArray{Element,Order}}

An `Order`-index tensor with a named, directed index attached to each leg.
Subtyping `AbstractArray` makes it transparent to `@tensor` contractions.

The backing store `D` is a free type parameter: `:native` mode uses plain
`Array{Element,Order}`; future backends may substitute other `AbstractArray`
subtypes without touching algorithm code.

    IndexedTensor(data, indices)

`data` must be an `AbstractArray` and `indices` an `NTuple` of `AbstractIndex`
values. The shared `Order` parameter in both types enforces that the number of
indices equals the array rank at the type level.
"""
struct IndexedTensor{Element, Order, D <: AbstractArray{Element, Order}} <: AbstractArray{Element, Order}
    data::D
    indices::NTuple{Order, AbstractIndex}
end

# ── AbstractArray interface ───────────────────────────────────────────────────

Base.size(t::IndexedTensor) = size(t.data)
Base.getindex(t::IndexedTensor, i...) = getindex(t.data, i...)
Base.setindex!(t::IndexedTensor, v, i...) = setindex!(t.data, v, i...)
Base.IndexStyle(::Type{<:IndexedTensor}) = IndexLinear()
Base.strides(t::IndexedTensor) = strides(t.data)

function Base.unsafe_convert(::Type{Ptr{Element}}, t::IndexedTensor{Element}) where {Element}
    return Base.unsafe_convert(Ptr{Element}, t.data)
end

# ── TensorOperations.jl interface ────────────────────────────────────────────

TensorOperations.tensorstructure(t::IndexedTensor, i::Int, ::Bool) = t.indices[i]

# ── Partition helpers ─────────────────────────────────────────────────────────

"""
    complement(p, A) -> Partition

Return the `Partition` of all indices in `A` that are not in `p`.
"""
function complement(p::Partition, A::IndexedTensor)
    remaining = filter(idx -> idx ∉ p.indices, collect(A.indices))
    return Partition(remaining)
end

"""
    bipartition(left, A) -> Bipartition

Construct a `Bipartition` pairing `left` with `complement(left, A)`.
"""
bipartition(left::Partition, A::IndexedTensor) = Bipartition(left, complement(left, A))

# ── group_legs ────────────────────────────────────────────────────────────────

function _resolve(idx::AbstractIndex, A::IndexedTensor)
    pos = findfirst(==(idx), A.indices)
    pos === nothing && throw(ArgumentError("index $idx not found in tensor"))
    return pos
end

"""
    group_legs(A, bp) -> IndexedTensor{El, 2}

Permute and reshape `A` into a 2-leg matrix according to `bp`. The left partition
becomes the row axis (a `MultiIx` over `bp.left.indices`) and the right partition
becomes the column axis. Frobenius norm is preserved exactly.

Throws `ArgumentError` if any partition index is absent from `A`, or if the
two partitions together do not cover all legs of `A`.
"""
function group_legs(A::IndexedTensor, bp::Bipartition)
    left_pos  = [_resolve(idx, A) for idx in bp.left.indices]
    right_pos = [_resolve(idx, A) for idx in bp.right.indices]
    perm      = vcat(left_pos, right_pos)

    if !(length(perm) == ndims(A) && isperm(perm))
        uncovered = [A.indices[i] for i in setdiff(1:ndims(A), perm)]
        msg = isempty(uncovered) ?
            "bipartition has duplicate index positions" :
            "bipartition covers $(length(unique(perm)))/$(ndims(A)) tensor indices — uncovered: $uncovered"
        throw(ArgumentError(msg))
    end

    left_dim  = prod(i -> size(A.data, i), left_pos;  init=1)
    right_dim = prod(i -> size(A.data, i), right_pos; init=1)
    M_data    = reshape(permutedims(A.data, perm), left_dim, right_dim)

    left_ix  = MultiIx(Tuple(bp.left.indices))
    right_ix = MultiIx(Tuple(bp.right.indices))
    return IndexedTensor(M_data, (left_ix, right_ix))
end
