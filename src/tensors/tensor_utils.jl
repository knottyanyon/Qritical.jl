# ── MulTIx auto-label helper & outer constructors ────────────────────────────

"""
    _autolabel(indices::Tuple{Vararg{AbstractIx}}) -> Symbol

Generate a label for a fused multi-index by concatenating the labels of its constituent
indices. Returns `:scalar` if the tuple is empty (a scalar has no indices to fuse).

This is an internal helper used by the `MulTIx` constructor to auto-generate labels when
none is provided. For example, `_autolabel((upper(:α, 3), lower(:σ, 2)))` returns `:ασ`.

# Examples

```jldoctest
julia> _autolabel((upper(:α, 3), lower(:σ, 2)))
:ασ

julia> _autolabel(())
:scalar
```
"""
function _autolabel(indices::Tuple{Vararg{AbstractIx}})
    isempty(indices) ? :scalar : Symbol(join(String.(label.(indices))))
end

MulTIx(indices::Tuple{Vararg{AbstractIx}}) = MulTIx(_autolabel(indices), indices)
MulTIx(indices::AbstractIx...) = MulTIx(indices)

# ── QTensor overloads of partition helpers ────────────────────────────────────
# These accept a QTensor as the second argument so callers don't have to
# extract A.indices manually. Defined here (after qtensor.jl) because they
# need both QTensor and the partition types.

"""
    complement(p::Partition, A::QTensor) -> Partition

Return the legs of `A` that are not in partition `p`, in the order they appear
in `A.indices`.  Delegates to `complement(p, A.indices)`.

# Examples
```jldoctest
julia> vL = upper(:vL, 2);  σ = upper(:σ, 3);  vR = lower(:vR, 4);

julia> A = QTensor(rand(2, 3, 4), (vL, σ, vR));

julia> complement(Partition([vL, σ]), A)
1-element Vector{AbstractIx}:
 TIx{Lower}(:vR, 4)
```
"""
complement(p::Partition, A::QTensor) = complement(p, A.indices)

"""
    bipartition(left::Partition, A::QTensor) -> Bipartition

Construct a [`Bipartition`](@ref) for tensor `A` whose right side is
`complement(left, A)`.

# Examples
```jldoctest
julia> vL = upper(:vL, 2);  σ = upper(:σ, 3);  vR = lower(:vR, 4);

julia> A = QTensor(rand(2, 3, 4), (vL, σ, vR));

julia> bp = bipartition(Partition([vL, σ]), A);

julia> bp.right[1] == vR
true
```
"""
bipartition(left::Partition, A::QTensor) = bipartition(left, A.indices)
