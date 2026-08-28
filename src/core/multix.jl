#=META
source:
  author: Bavithra
  coauthor:
  reviewer:
docstrings:
  author: Bavithra
  coauthor:
  reviewer:
refs:
credits:
=#

"""
    MulTIx <: AbstractIx

A fused index representing an **ordered** tuple of constituent
[`AbstractIx`](@ref) values.

`MulTIx` arises when several legs of a tensor are grouped into a single matrix
row or column before an SVD or contraction: for example, reshaping a rank-3
site tensor into a matrix for a bipartite SVD. Its `dim` is the product of the
constituent dimensions, matching the row/column count after reshaping.

The **order** of `indices` is significant: `(α, σ)` and `(σ, α)` correspond to
different permutations of the underlying array and are therefore not equal.

# Fields

  - `label   :: Symbol`                    — name of the fused leg
  - `indices :: Tuple{Vararg{AbstractIx}}` — constituent indices, in order

# Constructors

```julia
MulTIx(:fused, (α, σ))    # explicit label
MulTIx((α, σ))            #  hits MulTIx(::Tuple) → _autolabel → :ασ
MulTIx(α, σ)              # varargs sugar; same auto-label
```

# Examples

```jldoctest
julia> α = TIx(:α, 3);
       σ = TIx(:σ, 2);

julia> g = MulTIx(:ασ, (α, σ));

julia> dim(g)
6

julia> label(g)
:ασ

julia> g == MulTIx(:ασ, (σ, α))
false
```
"""
struct MulTIx <: AbstractIx
    label::Symbol    # relabeling as a composite index
    indices::Tuple{Vararg{AbstractIx}}   # immutable and ordered to ensure reshaping is correct
end

"""
    dim(g::MulTIx) -> Int

Return the total dimension of the fused index: the product of the dimensions of
all constituent indices. An empty `MulTIx` has `dim == 1` (empty product).

# Examples

```jldoctest
julia> α = TIx(:α, 3);
       σ = TIx(:σ, 2);

julia> dim(MulTIx(:g, (α, σ)))
6

julia> dim(MulTIx(:empty, ()))
1
```
"""
dim(g::MulTIx) = prod(dim, g.indices; init=1)   # composite leg dimension= product of the individual leg dimensions

"""
    label(g::MulTIx) -> Symbol

Return the label of the fused index.

# Examples

```jldoctest
julia> g = MulTIx(:ασ, (TIx(:α, 3), TIx(:σ, 2)));

julia> label(g)
:ασ
```
"""
label(g::MulTIx) = g.label

Base.:(==)(a::MulTIx, b::MulTIx) = a.label == b.label && a.indices == b.indices
Base.hash(g::MulTIx, h::UInt) = hash(g.label, hash(g.indices, h))
