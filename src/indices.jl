# ========================= Index location (variance)=========================

"""
    IxLoc

Abstract supertype for the variance of a tensor index.

Following the von Delft / tensor-network convention:

  - [`Upper`](@ref) — superscript, incoming arrow → **domain** (dual `V'`);
    the contravariant index of a ket-expansion *coefficient*, e.g. ``\\sigma``
    in ``|\\psi\\rangle = A^{\\sigma}|\\sigma\\rangle``
  - [`Lower`](@ref) — subscript, outgoing arrow → **codomain** (primal `V`);
    the covariant index carried by basis kets and bra-expansion coefficients

Variance is encoded as a **type parameter** rather than an enum value so that dispatch on `Upper` vs. `Lower` is resolved at compile time with zero runtime cost.

See also: [`TIx`](@ref), [`which_space`](@ref)
"""
abstract type IxLoc end

"""
    Upper <: IxLoc

Variance tag for an **upper** (superscript) index: incoming arrow, maps to the
**domain** of a `TensorMap` and to the dual space `V'`. This is the variance of
a contravariant ket-expansion coefficient index — the ``\\sigma`` of the stored
array ``A^{\\sigma}`` in ``|\\psi\\rangle = A^{\\sigma}|\\sigma\\rangle`` — so
physical legs of state tensors are `Upper`.

In Qritical.jl: `TIx{Upper}` satisfies `which_space == :domain`.

# Examples

```jldoctest
julia> which_space(TIx{Upper}(:α, 4))
:domain
```
"""
struct Upper <: IxLoc end

"""
    Lower <: IxLoc

Variance tag for a **lower** (subscript) index: outgoing arrow, maps to the
**codomain** of a `TensorMap` and to the primal space `V`. This is the variance
of a covariant index — the one labelling basis kets ``|\\sigma\\rangle`` or
carried by bra-expansion coefficients ``A^{\\dagger}_{\\sigma}``.

In Qritical.jl: `TIx{Lower}` satisfies `which_space == :codomain`.

# Examples

```jldoctest
julia> which_space(TIx{Lower}(:σ, 2))
:codomain
```
"""
struct Lower <: IxLoc end

# ========================= Abstract index interface =========================

"""
    AbstractIx

Root of the Qritical.jl index hierarchy.

Every concrete subtype must implement three methods that expose the index's identity and
structural role within a tensor:

  - [`dim(::AbstractIx)`](@ref)         — positive integer size of the index space.
    Required to check index compatibility during contraction and to reshape tensors for
    matrix operations (SVD, matrix-vector products).
  - [`label(::AbstractIx)`](@ref)       — symbolic name used for contraction matching.
    Two indices contract only if they have identical labels (one Upper, one Lower);
    the label is the key lookup in the contraction machinery.
  - [`which_space(::AbstractIx)`](@ref) — `:domain` for upper, `:codomain` for lower.
    Determines the variance semantics and which end of a `TensorMap` this leg maps to;
    essential for correct index order in matricisation (reshape for SVD) and for
    tracking adjoint operations.

Concrete subtypes: [`TIx`](@ref), [`MulTIx`](@ref).
"""
abstract type AbstractIx end

dim(i::AbstractIx) = error("`dim` not implemented for $(typeof(i))")
label(i::AbstractIx) = error("`label` not implemented for $(typeof(i))")
which_space(i::AbstractIx) = error("`which_space` not implemented for $(typeof(i))")

# ========================= TIx — elementary typed index =========================

"""
    TIx{L<:IxLoc} <: AbstractIx

A single tensor leg carrying a symbolic `label`, a positive `dim`ension, and
a **variance** encoded in the type parameter `L ∈ {Upper, Lower}`.

Variance is part of the index's identity: two `TIx` values with the same label
and dimension but *different* variance are not equal. This encodes the von Delft
convention — upper indices belong to the domain, lower indices to the codomain
— making contraction correctness a compile-time property. 
# Fields

  - `label :: Symbol` — name used for contraction matching (e.g. `:α`, `:σ_1`)
  - `dim   :: Int`    — strictly positive size of the index space (`dim ≥ 1`)

# Constructors

Prefer the convenience functions [`upper`](@ref) and [`lower`](@ref) over calling
`TIx{Upper}` / `TIx{Lower}` directly.

# Examples

```jldoctest
julia> α = TIx{Upper}(:α, 4);

julia> dim(α)
4

julia> label(α)
:α

julia> which_space(α)
:domain

julia> α == TIx{Upper}(:α, 4)
true

julia> α == TIx{Lower}(:α, 4)
false
```
"""
struct TIx{L<:IxLoc} <: AbstractIx

    label::Symbol
    dim::Int

    function TIx{L}(label::Symbol, dim::Int) where {L<:IxLoc}
        dim > 0 || throw(ArgumentError("TIx dim must be a positive integer, got $dim."))
        new{L}(label, dim)
    end
end

"""
    dim(ix::TIx) -> Int

Return the dimension of index `ix`.

# Examples

```jldoctest
julia> dim(TIx{Upper}(:α, 4))
4

julia> dim(TIx{Lower}(:σ, 2))
2
```
"""
dim(ix::TIx) = ix.dim

"""
    label(ix::TIx) -> Symbol

Return the symbolic label of index `ix`.

# Examples

```jldoctest
julia> label(TIx{Upper}(:α, 4))
:α

julia> label(TIx{Lower}(:σ_1, 2))
:σ_1
```
"""
label(ix::TIx) = ix.label

"""
    which_space(ix::TIx) -> Symbol

Return `:domain` for an [`Upper`](@ref) index and `:codomain` for a
[`Lower`](@ref) index, reflecting the von Delft convention.

The returned **space** is the Hilbert space that this index belongs to when viewed as a
leg of a `TensorMap`. In TensorKit parlance, a `TensorMap(V → W)` maps from domain `V`
(the input space) to codomain `W` (the output space). Each leg of the underlying dense
tensor falls into exactly one: Upper indices map to the domain (dual `V'`, contravariant),
while Lower indices map to the codomain (primal `W`, covariant). This correspondence
determines tensor contraction rules and the reshape order during matricisation for SVD:

  - upper = incoming arrow = contravariant coefficient index → `:domain` (dual `V'`)
  - lower = outgoing arrow = covariant basis index → `:codomain` (primal `V`)

# Examples

```jldoctest
julia> which_space(TIx{Upper}(:α, 4))
:domain

julia> which_space(TIx{Lower}(:σ, 2))
:codomain
```
"""
which_space(::TIx{Upper}) = :domain
which_space(::TIx{Lower}) = :codomain

# Extend Base.== and Base.hash so indices can be stored in dicts/sets and compared.
# Two indices are equal only if they have the same label, dimension, AND variance (Upper vs Lower).
# This is critical for contraction: we match Upper(label=:α, dim=2) with Lower(label=:α, dim=2),
# but not with Upper(label=:α, dim=2) or Upper(label=:β, dim=2). The hash enables fast index
# lookup when building contraction networks; dim is included so different dimensions don't
# collide, even if they happen to have the same label.
Base.:(==)(a::TIx{L}, b::TIx{L}) where {L<:IxLoc} = a.label == b.label && a.dim == b.dim
Base.:(==)(::TIx, ::TIx) = false   # different variance → never equal
Base.hash(ix::TIx, h::UInt) = hash(ix.label, hash(ix.dim, hash(typeof(ix), h)))

"""
    flip(ix::TIx) -> TIx

Raise or lower a single index: return an index with the same label and
dimension but the **opposite variance** (`Upper` ↔ `Lower`), moving the leg
between domain and codomain. Diagrammatically, `flip` reverses the leg's arrow.

Formally, raising and lowering an index corresponds to contracting with the metric tensor
``g_{ij}`` or its inverse ``g^{ij}``. In the orthonormal basis convention used throughout
Qritical, the metric is the identity (``g_{ij} = \\delta_{ij}``), so raising and lowering
are **purely syntactic**: they change the variance tag and the tensor map's leg placement
but do not alter numerical values. This is why `flip` only re-tags the index and is an
involution: `flip(flip(ix)) == ix`.

`flip` acts on **one leg only** — contrast with the adjoint, which flips *every*
leg, reverses their order, and conjugates the data. Flipping one end of a
contracted bond breaks the upper-with-lower pairing rule; a bond must be flipped
at **both** ends (as happens when the orthogonality centre moves across it).

See also: [`upper`](@ref), [`lower`](@ref), [`which_space`](@ref)

# Examples

```jldoctest
julia> flip(upper(:a, 3)) == lower(:a, 3)
true

julia> flip(flip(upper(:a, 3))) == upper(:a, 3)
true
```
"""
flip(ix::TIx{Upper}) = TIx{Lower}(ix.label, ix.dim)
flip(ix::TIx{Lower}) = TIx{Upper}(ix.label, ix.dim)

# ======================= Single-index convenience constructors ======================

"""
    upper(label::Symbol, dim::Int) -> TIx{Upper}

Construct an upper (domain) index. Prefer this over `TIx{Upper}(label, dim)`.

# Examples

```jldoctest
julia> α = upper(:α, 4);

julia> which_space(α)
:domain

julia> dim(α)
4
```
"""
upper(label::Symbol, dim::Int) = TIx{Upper}(label, dim)

"""
    lower(label::Symbol, dim::Int) -> TIx{Lower}

Construct a lower (codomain) index. Prefer this over `TIx{Lower}(label, dim)`.

# Examples

```jldoctest
julia> vR = lower(:vR, 2);

julia> which_space(vR)
:codomain

julia> dim(vR)
2
```
"""
lower(label::Symbol, dim::Int) = TIx{Lower}(label, dim)

# ========================= Batch constructors =========================

"""
    uppers(pairs::Pair{Symbol,Int}...) -> Tuple{TIx{Upper},...}

Construct a tuple of upper (domain) indices from `label => dim` pairs.

Calling with no arguments returns an empty tuple `()`, which is useful when
chaining with `filter` or `map` that may produce no results.

## Implementation

Uses Julia's broadcasting operator `.` to apply the constructor element-wise:
`first.(pairs)` extracts labels, `last.(pairs)` extracts dimensions, and
`TIx{Upper}.(...)` broadcasts the constructor to create each index.

# Examples

```jldoctest
julia> uppers()
()

julia> α, β = uppers(:α => 2, :β => 4);

julia> dim(α), dim(β)
(2, 4)

julia> which_space(β)
:domain
```
"""
uppers(pairs::Pair{Symbol,Int}...) = TIx{Upper}.(first.(pairs), last.(pairs))

"""
    lowers(pairs::Pair{Symbol,Int}...) -> Tuple{TIx{Lower},...}

Construct a tuple of lower (codomain) indices from `label => dim` pairs.

Calling with no arguments returns an empty tuple `()`.

# Examples

```jldoctest
julia> lowers()
()

julia> σ, τ = lowers(:σ => 2, :τ => 3);

julia> dim(σ), dim(τ)
(2, 3)

julia> which_space(τ)
:codomain
```
"""
lowers(pairs::Pair{Symbol,Int}...) = TIx{Lower}.(first.(pairs), last.(pairs))

"""
    uppers_range(base::Symbol, dim::Int, last::Int, start::Int=1)
        -> Tuple{TIx{Upper},...}

Construct a tuple of upper (domain) indices with auto-generated labels
`base_start`, `base_(start+1)`, …, `base_last`, all sharing the same `dim`.

Useful for building a sequence of bond or physical indices along a chain of
`L` sites — e.g. `uppers_range(:χ, D, L)` for `L` virtual bond legs of bond
dimension `D`.

# Arguments

  - `base`  — base symbol; the `i`-th label is `Symbol(base, :_, i)`
  - `dim`   — dimension shared by every index in the tuple
  - `last`  — upper bound of the range (inclusive)
  - `start` — lower bound of the range (default: `1`)

# Examples

```jldoctest
julia> r = uppers_range(:χ, 4, 3);

julia> length(r)
3

julia> label(r[1]), label(r[3])
(:χ_1, :χ_3)

julia> all(dim.(r) .== 4)
true

julia> r2 = uppers_range(:α, 2, 5, 3);

julia> length(r2)
3

julia> label(r2[1]), label(r2[3])
(:α_3, :α_5)
```
"""
function uppers_range(base::Symbol, dim::Int, last::Int, start::Int=1)
    Tuple(TIx{Upper}(Symbol(base, :_, i), dim) for i in start:last)
end

"""
    lowers_range(base::Symbol, dim::Int, last::Int, start::Int=1)
        -> Tuple{TIx{Lower},...}

Construct a tuple of lower (codomain) indices with auto-generated labels
`base_start`, `base_(start+1)`, …, `base_last`, all sharing the same `dim`.

# Arguments

  - `base`  — base symbol; the `i`-th label is `Symbol(base, :_, i)`
  - `dim`   — dimension shared by every index in the tuple
  - `last`  — upper bound of the range (inclusive)
  - `start` — lower bound of the range (default: `1`)

# Examples

```jldoctest
julia> r = lowers_range(:σ, 2, 4);

julia> length(r)
4

julia> label(r[1]), label(r[4])
(:σ_1, :σ_4)

julia> all(which_space.(r) .== Ref(:codomain))
true

julia> r2 = lowers_range(:β, 3, 6, 4);

julia> length(r2)
3

julia> label(r2[1]), label(r2[3])
(:β_4, :β_6)
```
"""
function lowers_range(base::Symbol, dim::Int, last::Int, start::Int=1)
    Tuple(TIx{Lower}(Symbol(base, :_, i), dim) for i in start:last)
end

# =============== MulTIx - grouped leg ========================================

"""
    MulTIx <: AbstractIx

A fused index representing an **ordered** tuple of constituent
[`AbstractIx`](@ref) values.

`MulTIx` arises when several legs of a tensor are grouped into a single matrix
row or column before an SVD or contraction — for example, reshaping a rank-3
site tensor into a matrix for a bipartite SVD. Its `dim` is the product of the
constituent dimensions, matching the row/column count after reshaping.

The **order** of `indices` is significant: `(α, σ)` and `(σ, α)` correspond to
different permutations of the underlying array and are therefore not equal.

!!! note "No single variance"

    Calling [`which_space`](@ref) on a `MulTIx` raises an error, because a
    grouped leg may combine upper and lower constituents and has no single
    well-defined variance.

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
julia> α = TIx{Upper}(:α, 3);
       σ = TIx{Lower}(:σ, 2);

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
    label::Symbol
    indices::Tuple{Vararg{AbstractIx}}
end

"""
    dim(g::MulTIx) -> Int

Return the total dimension of the fused index: the product of the dimensions of
all constituent indices. An empty `MulTIx` has `dim == 1` (empty product).

# Examples

```jldoctest
julia> α = TIx{Upper}(:α, 3);
       σ = TIx{Lower}(:σ, 2);

julia> dim(MulTIx(:g, (α, σ)))
6

julia> dim(MulTIx(:empty, ()))
1
```
"""
dim(g::MulTIx) = prod(dim, g.indices; init=1)

"""
    label(g::MulTIx) -> Symbol

Return the label of the fused index.

# Examples

```jldoctest
julia> g = MulTIx(:ασ, (TIx{Upper}(:α, 3), TIx{Lower}(:σ, 2)));

julia> label(g)
:ασ
```
"""
label(g::MulTIx) = g.label

Base.:(==)(a::MulTIx, b::MulTIx) = a.label == b.label && a.indices == b.indices
Base.hash(g::MulTIx, h::UInt) = hash(g.label, hash(g.indices, h))


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

# ========================= Partition & Bipartition ===========================

"""
    Partition

An ordered group of tensor legs, represented as `Vector{AbstractIx}`.

A `Partition` names the subset of a tensor's legs that will be collected
along one axis (rows or columns) when the tensor is matricised for an SVD or
contraction. The order within the partition determines the reshape order of
the underlying array.

Use [`Bipartition`](@ref) to pair two complementary `Partition`s, and
[`group_legs`](@ref) to apply the split to a `QTensor`.

# Examples

```jldoctest
julia> vL = upper(:vL, 2);
       σ = upper(:σ, 3);

julia> p = Partition([vL, σ]);

julia> length(p)
2

julia> p[1] == vL
true
```
"""
const Partition = Vector{AbstractIx}

"""
    Bipartition

A split of a tensor's legs into two ordered groups:

  - **`left`** — the legs that become the row axis (first index) after
    reshaping
  - **`right`** — the legs that become the column axis (second index) after
    reshaping

The constructor verifies that the two groups are **disjoint**: no single
`AbstractIx` value may appear in both `left` and `right`. Coverage — that the
two groups together account for every leg of the target tensor — is checked by
[`group_legs`](@ref) at the point of use, not here, because a `Bipartition`
may be built before a tensor is chosen.

# Fields

  - `left  :: Partition`
  - `right :: Partition`

# Examples

```jldoctest
julia> vL = upper(:vL, 2);
       σ = upper(:σ, 3);
       vR = lower(:vR, 4);

julia> bp = Bipartition(Partition([vL, σ]), Partition([vR]));

julia> length(bp.left)
2

julia> bp.right[1] == vR
true
```
"""
struct Bipartition
    left::Partition
    right::Partition

    function Bipartition(left::Partition, right::Partition)
        for ix in left
            ix ∈ right && throw(
                ArgumentError(
                    "Bipartition: leg '$(label(ix))' (dim=$(dim(ix))) appears in both " *
                    "the left and right partitions — each leg must belong to exactly one side.",
                ),
            )
        end
        new(left, right)
    end
end

"""
    complement(p::Partition, indices) -> Partition

Return the legs in `indices` that are **not** present in partition `p`,
preserving their original order.

`indices` may be any iterable of `AbstractIx` values — typically a tuple of
legs taken from a `QTensor` or a plain `Vector{AbstractIx}`.

Matching is by index equality (`==`), so two legs with the same label and
dimension but different variance (e.g. `upper(:σ, 2)` vs `lower(:σ, 2)`) are
treated as distinct.

See also: [`bipartition`](@ref), [`Bipartition`](@ref)

# Examples

```jldoctest
julia> vL = upper(:vL, 2);
       σ = upper(:σ, 3);
       vR = lower(:vR, 4);

julia> complement(Partition([vL, σ]), [vL, σ, vR])
1-element Vector{AbstractIx}:
 TIx{Lower}(:vR, 4)

julia> complement(Partition([]), [vL, σ])
2-element Vector{AbstractIx}:
 TIx{Upper}(:vL, 2)
 TIx{Upper}(:σ, 3)
```
"""
complement(p::Partition, indices) = Partition([ix for ix in indices if ix ∉ p])

"""
    bipartition(left::Partition, indices) -> Bipartition

Construct a [`Bipartition`](@ref) whose right side is automatically
`complement(left, indices)`.

This is the most convenient way to describe a Schmidt cut: name the legs you
want on the left (row) side and let the library fill in the right (column)
side.

# Examples

```jldoctest
julia> vL = upper(:vL, 2);
       σ = upper(:σ, 3);
       vR = lower(:vR, 4);

julia> bp = bipartition(Partition([vL, σ]), [vL, σ, vR]);

julia> bp.right[1] == vR
true
```
"""
bipartition(left::Partition, indices) = Bipartition(left, complement(left, indices))

# ========================= Bond label utility =================================

"""
    bond_label(base::Symbol, site::Int) -> Symbol

Generate a positional bond label by appending the site index to a base symbol.

Used throughout the MPS layer to create unique leg names for virtual bonds.
For example, the bond to the right of site ``i`` on a chain labelled `:χ` is
`bond_label(:χ, i)` ``= \\texttt{:χ}_i``.

# Examples

```jldoctest
julia> bond_label(:χ, 3)
:χ3

julia> bond_label(:α, 12)
:α12
```
"""
bond_label(base::Symbol, site::Int) = Symbol(base, site)