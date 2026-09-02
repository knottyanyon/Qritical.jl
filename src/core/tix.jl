#=META
source:
  author: Bavithra
  coauthor: N/A
  reviewer:
docstrings:
  author: Bavithra
  coauthor:
  reviewer:
refs:
credits: [ITensor.jl](https://github.com/ITensor/ITensors.jl)
=#

using TensorKit

"""
    AbstractIx

Root of the Qritical.jl index hierarchy.

Every concrete subtype must implement two methods that expose the index's identity within a
tensor:

  - [`dim(::AbstractIx)`](@ref)   - the (possibly non-integer, for anyonic sectors) size of
    the index space. Required to check index compatibility during contraction and to reshape
    tensors for matrix operations (SVD, matrix-vector products).
  - [`space(::AbstractIx)`](@ref) - the underlying `TensorKit.ElementarySpace` this index
    represents, carrying its full symmetry-sector structure (trivial `ComplexSpace` or a
    graded space). `dim` is always derivable from `space` via `TensorKit.dim`.

[`label(::AbstractIx)`](@ref) is **not** part of the required interface: TensorKit's own
`ElementarySpace`/`TensorMap` model has no concept of a named leg at all (leg matching there
is purely positional, by codomain/domain order), so naming is opt-in per subtype rather than
universal. [`MulTIx`](@ref) implements `label` for its own identity; [`TIx`](@ref) deliberately
does not.

Concrete subtypes: [`TIx`](@ref), [`MulTIx`](@ref).
"""
abstract type AbstractIx end   # root of the index hierarchy

# abstract interface stubs
dim(i::AbstractIx) = error("`dim` not implemented for $(typeof(i))")
label(i::AbstractIx) = error("`label` not implemented for $(typeof(i))")
space(i::AbstractIx) = error("`space` not implemented for $(typeof(i))")

# SECTION -  TIx: a single leg, directly wrapping a TensorKit space

"""
    TIx{S<:TensorKit.ElementarySpace,L<:LegRole} <: AbstractIx

A single tensor leg, represented directly as a `TensorKit.ElementarySpace` - the same thing
TensorKit itself uses for a leg (`ComplexSpace` for the trivial/no-symmetry case, or a
`GradedSpace{I<:Sector,D}` carrying real sector => multiplicity data for a symmetric leg) -
together with the [`LegRole`](@ref) it plays ([`PhysicalLeg`](@ref) or [`VirtualLeg`](@ref)).
Both parameters are carried in the type so [`symmetry_structure`](@ref) and
[`entanglement_structure`](@ref) dispatch at compile time, in the same "Holy traits" style as
[`graph_trait`](@ref).

`TIx` carries **no label**: TensorKit's own model has no concept of a named leg anywhere
(`VectorSpace`/`ElementarySpace`/`GradedSpace`/`ProductSpace`/`TensorMap` are all purely
positional - codomain/domain order only, confirmed by inspecting TensorKit v0.17.1's source),
so `TIx` matches that model exactly rather than layering a Qritical-specific naming scheme on
top of it. Calling `label(::TIx)` is a deliberate error (via `AbstractIx`'s interface stub),
not a missing method - it signals "this index has no name," not an oversight.

# Fields

Both fields are carried as compile-time type parameters (`TIx{S,L}`), not just runtime field
types - this is what lets `symmetry_structure`/`entanglement_structure` dispatch on `Type{TIx}`
alone, with no instance needed.

$(Glossaries.Field{Core}()([:space, :leg]))

`leg` defaults to [`PhysicalLeg`](@ref) unless constructed otherwise (see below).

# Constructors

  - `TIx(space::TensorKit.ElementarySpace, leg::LegRole=PhysicalLeg())` - wrap an existing space
    directly; this is how a `TIx` carries real symmetry-sector information (pass a
    `GradedSpace`), and how it's marked as a [`VirtualLeg`](@ref) (pass `VirtualLeg()`).
  - `TIx(d::Int, leg::LegRole=PhysicalLeg())` - convenience constructor for the common trivial-sector
    case, equivalent to `TIx(ComplexSpace(d), leg)`.

# Examples

```jldoctest
julia> α = TIx(4);

julia> dim(α)
4

julia> α == TIx(4)
true

julia> α == TIx(5)
false

julia> g = TIx(GradedSpace(Z2Irrep(0) => 2, Z2Irrep(1) => 3));

julia> dim(g)
5

julia> χ = TIx(8, VirtualLeg());

julia> entanglement_structure(typeof(χ))
CarriesEntanglementInfo()
```
"""
struct TIx{S<:TensorKit.ElementarySpace,L<:LegRole} <: AbstractIx
    space::S
    leg::L
end
TIx(space::TensorKit.ElementarySpace) = TIx(space, PhysicalLeg())
function TIx(d::Int, leg::LegRole=PhysicalLeg())   # convenience constructor for the trivial (no-symmetry) case
    # TensorKit.ComplexSpace itself does not validate d > 0 (ComplexSpace(-1) silently
    # succeeds with dim -1), so this check has to live here, not be inherited for free.
    d > 0 || throw(ArgumentError("TIx dim must be a positive integer, got $d."))
    return TIx(TensorKit.ComplexSpace(d), leg)
end

"""
    symmetry_structure(::Type{TIx{S,L}}) where {S,L} -> SymmetryStructure

The [`SymmetryStructure`](@ref) carried by a `TIx` type, resolved purely from its space type
parameter `S` (compile-time dispatch, independent of `L`).
"""
symmetry_structure(::Type{TIx{S,L}}) where {S,L} = symmetry_structure(S)

"""
    entanglement_structure(::Type{TIx{S,L}}) where {S,L} -> EntanglementStructure

The [`EntanglementStructure`](@ref) carried by a `TIx` type, resolved purely from its leg type
parameter `L` (compile-time dispatch, independent of `S`).
"""
entanglement_structure(::Type{TIx{S,L}}) where {S,L} = entanglement_structure(L)

# Instance-level forwarding so carries_symmetry_info/carries_entanglement_info (which call
# symmetry_structure/entanglement_structure on the instance, matching MulTIx's own instance-level
# methods below) work uniformly for TIx too, on top of the Type-based dispatch above.
symmetry_structure(ix::TIx) = symmetry_structure(typeof(ix))
entanglement_structure(ix::TIx) = entanglement_structure(typeof(ix))

"""
    dim(ix::TIx) -> Int

Return the dimension of index `ix`, delegating to `TensorKit.dim(ix.space)`. This is always
an `Int` for group-like sectors (including the trivial case) but can be non-integer for
anyonic sectors (e.g. `dim(σ) = √2` for the Ising anyon) - `TensorKit.dim` already handles this
correctly per sector type, so `TIx` inherits that behavior for free.

# Examples

```jldoctest
julia> dim(TIx(4))
4

julia> dim(TIx(2))
2
```
"""
dim(ix::TIx) = TensorKit.dim(ix.space)

"""
    space(ix::TIx) -> TensorKit.ElementarySpace

Return the underlying `TensorKit.ElementarySpace` of index `ix`.

# Examples

```jldoctest
julia> space(TIx(4))
ℂ^4
```
"""
space(ix::TIx) = ix.space

# Extend Base.== and Base.hash so indices can be stored in dicts/sets and compared.
# Two TIx values are equal iff their underlying spaces are equal (TensorKit's own
# ElementarySpace subtypes already define == / hash correctly per sector structure,
# so this delegates rather than reimplementing space comparison).
Base.:(==)(a::TIx, b::TIx) = a.space == b.space && a.leg == b.leg
Base.hash(ix::TIx, h::UInt) = hash(ix.leg, hash(ix.space, h))

# SECTION -  MulTIx: a fused leg, correctly space-fused via TensorKit

"""
    MulTIx <: AbstractIx

A fused index representing an **ordered** tuple of constituent [`AbstractIx`](@ref) values.

`MulTIx` arises when several legs of a tensor are grouped into a single matrix row or column
before an SVD or contraction: for example, reshaping a rank-3 site tensor into a matrix for a
bipartite SVD. Its [`space`](@ref) is the genuine `TensorKit.fuse` of the constituents' own
spaces, not a flat product of their dimensions - for constituents carrying real sector
information, fusion combines sector multiplicities via the sectors' own fusion rules
(`⊗`/`Nsymbol`), which is not the same as multiplying total dimensions once symmetry is
involved. [`dim`](@ref) is then just `TensorKit.dim(space(g))`, correct in both the trivial
and symmetric cases (for trivial `ComplexSpace` constituents, `fuse` reduces to the familiar
dimension product). An empty `MulTIx` (no constituents) has `space` equal to the trivial
1-dimensional space `ComplexSpace(1)` (the monoidal unit), matching `dim == 1`, since
`TensorKit.fuse` itself has no defined behavior for zero arguments.

The **order** of `indices` is significant: `(α, σ)` and `(σ, α)` correspond to different
permutations of the underlying array and are therefore not equal.

# Fields

  - `label   :: Symbol`                    - name of the fused leg (MulTIx's own identity;
    unrelated to whether the constituent `TIx` values carry labels - they don't)
  - `indices :: Tuple{Vararg{AbstractIx}}` - constituent indices, in order

$(Glossaries.Field{Core}()([:leg]))

`leg` defaults to [`VirtualLeg`](@ref) unless constructed otherwise, since a `MulTIx` is
normally formed right before a bipartite SVD.

# Examples

```jldoctest
julia> α = TIx(3);
       σ = TIx(2);

julia> g = MulTIx(:ασ, (α, σ));

julia> dim(g)
6

julia> label(g)
:ασ

julia> g == MulTIx(:ασ, (σ, α))
false

julia> a = TIx(GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1));
       b = TIx(GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1));

julia> space(MulTIx(:ab, (a, b)))   # real sector fusion, not a flat 4-dim block
Rep[ℤ₂](0 => 2, 1 => 2)
```
"""
struct MulTIx <: AbstractIx
    label::Symbol
    indices::Tuple{Vararg{AbstractIx}}
    leg::LegRole
end
function MulTIx(label::Symbol, indices::Tuple{Vararg{AbstractIx}})
    return MulTIx(label, indices, VirtualLeg())
end

"""
    symmetry_structure(g::MulTIx) -> SymmetryStructure

The [`SymmetryStructure`](@ref) carried by `g`, resolved from its fused `space(g)` at the
instance level (the fused space depends on the runtime `indices` tuple, so unlike [`TIx`](@ref)
this can't be a pure `Type{MulTIx}` dispatch).
"""
symmetry_structure(g::MulTIx) = symmetry_structure(typeof(space(g)))

"""
    entanglement_structure(g::MulTIx) -> EntanglementStructure

The [`EntanglementStructure`](@ref) carried by `g`, resolved from its `leg` field.
"""
entanglement_structure(g::MulTIx) = entanglement_structure(typeof(g.leg))

"""
    space(g::MulTIx) -> TensorKit.ElementarySpace

Return the fused `TensorKit.ElementarySpace` of `g`'s constituents, via `TensorKit.fuse`
(recursing correctly if a constituent is itself a `MulTIx`, since `space` is defined for both).
An empty `MulTIx` returns the trivial space `ComplexSpace(1)`, since `TensorKit.fuse` has no
zero-argument method.

# Examples

```jldoctest
julia> space(MulTIx(:empty, ()))
ℂ^1

julia> space(MulTIx(:ασ, (TIx(3), TIx(2))))
ℂ^6
```
"""
function space(g::MulTIx)
    return if isempty(g.indices)
        TensorKit.ComplexSpace(1)
    else
        TensorKit.fuse(space.(g.indices)...)
    end
end

"""
    dim(g::MulTIx) -> Int

Return the dimension of the fused index: `TensorKit.dim(space(g))`. An empty `MulTIx` has
`dim == 1` (the trivial space's dimension).

# Examples

```jldoctest
julia> α = TIx(3);
       σ = TIx(2);

julia> dim(MulTIx(:g, (α, σ)))
6

julia> dim(MulTIx(:empty, ()))
1
```
"""
dim(g::MulTIx) = TensorKit.dim(space(g))

"""
    label(g::MulTIx) -> Symbol

Return the label of the fused index.

# Examples

```jldoctest
julia> g = MulTIx(:ασ, (TIx(3), TIx(2)));

julia> label(g)
:ασ
```
"""
label(g::MulTIx) = g.label

function Base.:(==)(a::MulTIx, b::MulTIx)
    return a.label == b.label && a.indices == b.indices && a.leg == b.leg
end
Base.hash(g::MulTIx, h::UInt) = hash(g.leg, hash(g.label, hash(g.indices, h)))
