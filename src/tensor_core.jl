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
