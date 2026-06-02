# ── AbstractIndex ─────────────────────────────────────────────────────────────

"""
    AbstractIndex

Root of the index hierarchy. Every concrete subtype must implement `ndim`.
"""
abstract type AbstractIndex end

"""
    ndim(i::AbstractIndex) -> Int

Return the number of distinct values the index `i` can take (its dimension).

# Examples
```jldoctest
julia> ndim(upper(:σ, 2))
2

julia> ndim(lower(:α, 4))
4

julia> ndim(MultiIx(:αβ, (upper(:α, 2), lower(:β, 3))))
6
```
"""
ndim(i::AbstractIndex) = error("ndim not implemented for $(typeof(i))")

# ── IndexLoc ─────────────────────────────────────────────────────────────────

"""
    IndexLoc

Abstract type for the position of a covariant index in tensor notation.

Concrete subtypes:
- `Upper` — superscript position (contravariant; incoming arrow in TN diagrams)
- `Lower` — subscript  position (covariant;  outgoing arrow in TN diagrams)

Position is encoded as a type parameter rather than an enum value so that
dispatch on index location is resolved at compile time.
"""
abstract type IndexLoc end

"""Superscript / upper index position."""
struct Upper <: IndexLoc end

"""Subscript / lower index position."""
struct Lower <: IndexLoc end

# ── TIx ──────────────────────────────────────────────────────────────────

"""
    TIx{L <: IndexLoc}

A named, positioned tensor index. `L` is either `Upper` or `Lower`.

`TIx{Upper}` and `TIx{Lower}` are distinct types, so a pair of
same-label indices with opposite positions can be matched purely by the type
system — no runtime comparison needed.

# Examples
```jldoctest
julia> TIx{Upper}(:σ, 2)
TIx{Upper}(:σ, 2)

julia> TIx{Lower}(:α, 4)
TIx{Lower}(:α, 4)

julia> TIx{Upper}(:α, 4) == TIx{Lower}(:α, 4)   # position is part of the type
false

julia> TIx{Upper}(:α, 4) == TIx{Upper}(:α, 4)
true

julia> ndim(TIx{Upper}(:vL, 1))   # boundary bond of a finite MPS
1

julia> TIx{Upper}(:α, 0)
ERROR: ArgumentError: TIx ndim must be positive, got 0
[...]
```
"""
struct TIx{L<:IndexLoc} <: AbstractIndex
    label::Symbol
    ndim::Int
    function TIx{L}(label::Symbol, ndim::Int) where {L<:IndexLoc}
        ndim > 0 || throw(ArgumentError("TIx ndim must be positive, got $ndim"))
        new{L}(label, ndim)
    end
end

ndim(i::TIx) = i.ndim

"""
    upper(label, ndim) → TIx{Upper}
    lower(label, ndim) → TIx{Lower}

Construct a single named index at the given position.

# Examples
```jldoctest
julia> upper(:σ, 2)
TIx{Upper}(:σ, 2)

julia> lower(:α, 4)
TIx{Lower}(:α, 4)

julia> upper(:σ, 2) == TIx{Upper}(:σ, 2)
true
```
"""
upper(label::Symbol, ndim::Int) = TIx{Upper}(label, ndim)
lower(label::Symbol, ndim::Int) = TIx{Lower}(label, ndim)

"""
    uppers(label => ndim, ...) → Tuple of TIx{Upper}
    lowers(label => ndim, ...) → Tuple of TIx{Lower}

Construct several indices of the same position in one call using `label => ndim`
pairs. Returns a `Tuple` that can be destructured directly.

# Examples
```jldoctest
julia> vL, vR = uppers(:vL => 4, :vR => 4)
(TIx{Upper}(:vL, 4), TIx{Upper}(:vR, 4))

julia> ndim(vL)
4

julia> (σ,) = lowers(:σ => 2)
(TIx{Lower}(:σ, 2),)
```
"""
uppers(pairs::Pair{Symbol,Int}...) = Tuple(TIx{Upper}(p.first, p.second) for p in pairs)
lowers(pairs::Pair{Symbol,Int}...) = Tuple(TIx{Lower}(p.first, p.second) for p in pairs)

# ── MultiIx ──────────────────────────────────────────────────────────────

"""
    MultiIx(label, indices)

A composite index formed by bundling several constituent `AbstractIndex` values.
Used to reshape a high-order tensor into a matrix for SVD: one `MultiIx`
per side of the bisection.

The dimension of a `MultiIx` is the product of its constituent dimensions
because it enumerates all combinations of the constituent values.

# Examples
```jldoctest
julia> α, β = upper(:α, 2), lower(:β, 3);

julia> g = MultiIx(:αβ, (α, β))
MultiIx(:αβ, (TIx{Upper}(:α, 2), TIx{Lower}(:β, 3)))

julia> ndim(g)
6

julia> ndim(MultiIx(:single, (upper(:γ, 5),)))   # one constituent passes through
5

julia> ndim(MultiIx(:empty, ()))   # empty product: no legs, scalar dimension 1
1
```
"""
struct MultiIx <: AbstractIndex
    label::Symbol
    indices::Tuple{Vararg{AbstractIndex}}
end

# The dimension of a grouped index is the total number of distinct value
# combinations across all constituents. Since the constituent indices are
# independent, this equals the product of their individual ndims.
#
#
ndim(g::MultiIx) = prod(ndim, g.indices; init=1)
