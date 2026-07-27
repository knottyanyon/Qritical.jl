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
function _autolabel(indices::Tuple{Vararg{AbstractIx}})   # `Tuple{Vararg{AbstractIx}}` = a Tuple of any number of AbstractIx elements; `Vararg` = variadic
    isempty(indices) ? :scalar : Symbol(join(String.(label.(indices))))   # ternary: if empty return `:scalar`; else: `label.(indices)` = broadcast `label` to get a tuple of Symbols; `String.(...)` = broadcast `String` to convert each Symbol to a String; `join(...)` = concatenate all strings; `Symbol(...)` = convert the joined string back to a Symbol
end

MulTIx(indices::Tuple{Vararg{AbstractIx}}) = MulTIx(_autolabel(indices), indices)   # outer constructor: takes a Tuple of indices, auto-generates the label, then calls the full 2-argument struct constructor; Python: `@classmethod` or `__init__` with optional `label` argument
MulTIx(indices::AbstractIx...) = MulTIx(indices)   # varargs outer constructor: `f(x...)` collects arguments into a Tuple ; delegates to the Tuple form above

# ── QTensor overloads of partition helpers ────────────────────────────────────
# These accept a QTensor as the second argument so callers don't have to
# extract A.indices manually. Defined here (after qtensor.jl) because they
# need both QTensor and the partition types.

"""
    complement(p::Partition, A::QTensor) -> Partition

Return the legs of `A` that are not in partition `p`, in the order they appear
in `A.indices`.  Delegates to `complement(p, A.indices)`.

See also: [`complement(::Partition, indices)`](@ref)

# Examples

```jldoctest
julia> vL = upper(:vL, 2);
       σ = upper(:σ, 3);
       vR = lower(:vR, 4);

julia> A = QTensor(rand(2, 3, 4), (vL, σ, vR));

julia> complement(Partition([vL, σ]), A)
1-element Vector{AbstractIx}:
 TIx{Lower}(:vR, 4)   # QTensor overload: extracts A.indices and delegates to the generic `complement(p, indices)` method; avoids forcing callers to write `complement(p, A.indices)` manually
```
"""
complement(p::Partition, A::QTensor) = complement(p, A.indices)   # QTensor overload: extracts A.indices and delegates to the generic `complement(p, indices)` method; avoids forcing callers to write `complement(p, A.indices)` manually

"""
    bipartition(left::Partition, A::QTensor) -> Bipartition

Construct a [`Bipartition`](@ref) for tensor `A` whose right side is
`complement(left, A)`.

# Examples

```jldoctest
julia> vL = upper(:vL, 2);
       σ = upper(:σ, 3);
       vR = lower(:vR, 4);

julia> A = QTensor(rand(2, 3, 4), (vL, σ, vR));

julia> bp = bipartition(Partition([vL, σ]), A);

julia> bp.right[1] == vR
true   # QTensor overload: extracts A.indices and delegates to `bipartition(left, indices)` from partition.jl; the right side is automatically the complement
```
"""
bipartition(left::Partition, A::QTensor) = bipartition(left, A.indices)   # QTensor overload: extracts A.indices and delegates to `bipartition(left, indices)` from partition.jl; the right side is automatically the complement

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
bond_label(base::Symbol, site::Int) = Symbol(base, site)   # `Symbol(base, site)` = concatenate base Symbol with integer site to create e.g. :χ3; Python has no Symbol type but this is analogous to f":χ{site}" as an interned string; used to build unique names for each bond in the MPS chain
