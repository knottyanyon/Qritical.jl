using LinearAlgebra
using TensorOperations

# Naming deviates from Julia convention ({T,N}) intentionally:
#   Element — the scalar field (e.g. ComplexF64), not constrained to <:Number.
#             
#   Order   — number of indices; we reserve "rank" for the SVD / linear-algebra
#             sense
"""
    IndexedTensor{Element, Order, D<:AbstractArray{Element,Order}}

An `Order`-index tensor with a named, covariant/contravariant index attached to each leg.
Subtyping `AbstractArray` makes it transparent to `@tensor` contractions.

The backing store `D` is a free type parameter: `:native` mode uses plain
`Array{Element,Order}` from Julia Base; future backends may substitute other `AbstractArray`
subtypes without touching algorithm code. The goal is facilitate the use of efficient sparse array descriptions if one wishes to implement symmetry details to take advantage of the block structure of the  hamiltonians that we usually deal with. 

    IndexedTensor(data, indices)

`data` must be an `AbstractArray` encoding the numerical data of the tensor and `indices` an `NTuple` of `AbstractIndex`
values. The shared `Order` parameter in both types enforces that the number of
indices equals the array rank at the type level.
"""
struct IndexedTensor{Element,Order,D<:AbstractArray{Element,Order}} <:
       AbstractArray{Element,Order}
    data::D
    indices::NTuple{Order,AbstractIndex}
end

# ── AbstractArray interface ───────────────────────────────────────────────────
# All data operations delegate to `t.data`; `IndexedTensor` adds no storage
# overhead and is fully transparent to Julia's array machinery, broadcasting,
# and `@tensor` contractions.

Base.size(t::IndexedTensor) = size(t.data)
Base.getindex(t::IndexedTensor, i...) = getindex(t.data, i...)
Base.setindex!(t::IndexedTensor, v, i...) = setindex!(t.data, v, i...)

"""
    Base.IndexStyle(::Type{<:IndexedTensor})

Declare linear indexing for `IndexedTensor`.

`IndexLinear()` tells Julia that elements are cheaply accessible via a single
integer offset, which enables loop fusion, `@simd` vectorisation, and avoids
the Cartesian-to-linear conversion overhead in inner loops. The choice is valid
because the backing store `D` is always a strided array with contiguous linear
storage (plain `Array` in `:native` mode).
"""
Base.IndexStyle(::Type{<:IndexedTensor}) = IndexLinear()

"""
    Base.strides(t::IndexedTensor)

Return the memory strides of the backing array.

Required alongside `unsafe_convert` so that BLAS/LAPACK routines (and any
external C library expecting a raw pointer + stride descriptor) can operate
directly on `IndexedTensor` data without an intermediate copy.
"""
Base.strides(t::IndexedTensor) = strides(t.data)

"""
    Base.unsafe_convert(::Type{Ptr{Element}}, t::IndexedTensor{Element})

Return a raw pointer to the first element of the backing array.

Together with `strides`, this makes `IndexedTensor` compatible with BLAS/LAPACK
dispatch (e.g. via `LinearAlgebra.BLAS.gemm!`) and with AD frameworks such as
Enzyme that operate on raw memory. The `unsafe` prefix is Julia's convention:
the caller is responsible for ensuring the array is not garbage-collected and
that accesses stay within bounds.
"""
function Base.unsafe_convert(
    ::Type{Ptr{Element}}, t::IndexedTensor{Element}
) where {Element}
    return Base.unsafe_convert(Ptr{Element}, t.data)
end

# ── TensorOperations.jl interface ────────────────────────────────────────────

"""
    TensorOperations.tensorstructure(t::IndexedTensor, i::Int, conjA::Bool) -> AbstractIndex

Return the `AbstractIndex` metadata for leg `i` of tensor `t`, as required by the
`TensorOperations.jl` contraction interface.

`TensorOperations.jl` calls this function during allocation and compatibility checking
when building a `@tensor` contraction. For a plain `AbstractArray` the default returns
`size(A, i)` — just an integer. Overriding it to return the full `AbstractIndex` exposes
the label, direction (`Upper`/`Lower`), and dimension to the contraction engine, enabling
future leg-compatibility checks beyond a bare dimension match.

The `conjA` flag (`true` when the tensor appears conjugated in a `@tensor` expression)
is forwarded by the interface but does not affect index identity — the same `AbstractIndex`
is returned regardless.
"""
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

"""
    _resolve(idx, A) -> Int

Return the dimension position of `idx` in `A.indices`.

Equality is tested with `==`, which for `TIx` requires matching label, `ndim`,
*and* `Upper`/`Lower` position — two indices with the same label but opposite
covariance are distinct and will not match. This is intentional: a leg going
in and a leg going out are different objects even if they share a name.

Throws `ArgumentError` if `idx` is not found, so callers (`group_legs`) get a
legible message rather than a `nothing`-dereference.
"""
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
    left_pos = [_resolve(idx, A) for idx in bp.left.indices]
    right_pos = [_resolve(idx, A) for idx in bp.right.indices]
    perm = vcat(left_pos, right_pos)

    if !(length(perm) == ndims(A) && isperm(perm))
        uncovered = [A.indices[i] for i in setdiff(1:ndims(A), perm)]
        msg = if isempty(uncovered)
            "bipartition has duplicate index positions"
        else
            "bipartition covers $(length(unique(perm)))/$(ndims(A)) tensor indices — uncovered: $uncovered"
        end
        throw(ArgumentError(msg))
    end

    left_dim = prod(i -> size(A.data, i), left_pos; init=1)
    right_dim = prod(i -> size(A.data, i), right_pos; init=1)
    M_data = reshape(permutedims(A.data, perm), left_dim, right_dim)

    left_ix = MultiIx(Tuple(bp.left.indices))
    right_ix = MultiIx(Tuple(bp.right.indices))
    return IndexedTensor(M_data, (left_ix, right_ix))
end
