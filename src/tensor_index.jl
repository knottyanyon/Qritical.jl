# Tensor index machinery for covariant notation.
#
# Design is inspired by SymPy's TensorIndex / TensorIndexType and by the
# hand-calculation conventions in Altland & von Delft (2019). The priority
# here is to make the index algebra match the notation on paper.

# ── IndexDirection ────────────────────────────────────────────────────────────

"""
    IndexDirection

Direction of a tensor index in covariant notation, corresponding to the index
position (superscript vs subscript) and its arrow direction in tensor network
diagrams:

| Index position    | Arrow in diagram | Value       |
|:------------------|:-----------------|:------------|
| Superscript (up)  | incoming → •     | `UpIndex`   |
| Subscript (down)  | outgoing • →     | `DownIndex` |

Contractions must always pair one `UpIndex` index with one `DownIndex` index.
"""
@enum IndexDirection UpIndex DownIndex

"""
    flip(dir::IndexDirection)

Raise or lower the index.
"""
flip(dir::IndexDirection) = dir == UpIndex ? DownIndex : UpIndex

# ── AbstractIndex and concrete subtypes ──────────────────────────────────────

"""
    AbstractIndex

Supertype for all tensor index/leg types. Concrete subtypes: `PhysicalIndex`,
`BondIndex`.
"""
abstract type AbstractIndex end

"""
    PhysicalIndex(label, site, dir)

An index representing the local Hilbert space at a lattice `site`.
`dir` is `UpIndex` (superscript) or `DownIndex` (subscript). The local
Hilbert space dimension is computed from `site` via `local_hilbert_dim`.
"""
struct PhysicalIndex <: AbstractIndex
    label::Symbol
    site::AbstractSite
    dir::IndexDirection
end

"""
    BondIndex(label, from, to, dim, dir)

A virtual/bond index carrying entanglement between two tensors. `from` and `to`
are the ordinals of the source and target sites of the bond arrow. `dim` is the
bond dimension. `dir` encodes which role this slot plays on the tensor that
holds it, and must be consistent with the site:

| Site   | Expected `dir` | Meaning                                       |
|:-------|:---------------|:----------------------------------------------|
| `from` | `DownIndex`    | arrow leaves this tensor (outgoing, subscript) |
| `to`   | `UpIndex`      | arrow enters this tensor (incoming, superscript) |

For a bond α with arrow from site 2 to site 3:
- slot on site 2: `BondIndex(:α, 2, 3, D, DownIndex)` (outgoing)
- slot on site 3: `BondIndex(:α, 2, 3, D, UpIndex)`   (incoming)
"""
struct BondIndex <: AbstractIndex
    label::Symbol
    from::Int
    to::Int
    dim::Int
    dir::IndexDirection
end

is_physical(::PhysicalIndex) = true
is_physical(::BondIndex) = false
is_bond(i::AbstractIndex) = !is_physical(i)


"""
    local_hilbert_dim(i::PhysicalIndex)
    local_hilbert_dim(i::BondIndex)

Return the local Hilbert space dimension of index `i`.

For a `PhysicalIndex` the dimension is delegated to the site:
`local_hilbert_dim(i.site)`. For a `BondIndex` it is the stored bond dimension.

```jldoctest
julia> local_hilbert_dim(PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex))   # spin-1/2: 2S+1 = 2
2

julia> local_hilbert_dim(PhysicalIndex(:n, SpinlessBosonicSite(1; n_max_occ=3), DownIndex))  # |0⟩…|3⟩: dim = 4
4

julia> local_hilbert_dim(BondIndex(:α, 16, DownIndex))   # stored directly, no site involved
16
```
"""
local_hilbert_dim(i::PhysicalIndex) = local_hilbert_dim(i.site)
local_hilbert_dim(i::BondIndex) = i.dim  # bond dimension is a free parameter, not derived from a site type


# ── dual, adjoint, directional helpers ───────────────────────────────────────

"""
    dual(i::AbstractIndex)

Return a copy of `i` with the `IndexDirection` flipped. Also available as `i'`
via `Base.adjoint`.
"""
dual(i::PhysicalIndex) = PhysicalIndex(i.label, i.site, flip(i.dir))
dual(i::BondIndex) = BondIndex(i.label, i.from, i.to, i.dim, flip(i.dir))

Base.adjoint(i::AbstractIndex) = dual(i)

"""
    isdual(i, j)

Return `true` if `i` and `j` form a valid Einstein contraction pair (analogous
to xTensor's `PairQ[a, -a]`): same kind, same label, same site/bond-location,
same dimension, and opposite direction.
"""
isdual(i::BondIndex, j::BondIndex) =
    i.dir != j.dir && i.label == j.label && i.from == j.from && i.to == j.to && i.dim == j.dim

"""
    isdual(i::PhysicalIndex, j::PhysicalIndex)

Return `true` if `i` and `j` form a valid Einstein pair for a physical leg:
same label, same site, opposite direction.

Design note — "same site" admits two interpretations:
- Reference identity (`i.site === j.site`): strictest; requires reusing the exact
  same site object. Safe in practice since sites are typically constructed once
  and stored in a lattice array.
- Structural equality (`typeof(i.site) == typeof(j.site) && i.site.lattice_ordinal == j.site.lattice_ordinal`):
  allows two independently-constructed sites with the same type and ordinal to
  match. More flexible, but permits subtle aliasing between site types that share
  a lattice ordinal.
"""
function isdual(i::PhysicalIndex, j::PhysicalIndex)
    i.dir != j.dir && i.label == j.label && i.site === j.site
end

isdual(::AbstractIndex, ::AbstractIndex) = false

"""
    as_down(i) → index with `dir == DownIndex` (idempotent)
    as_up(i)   → index with `dir == UpIndex`   (idempotent)
"""
as_down(i::AbstractIndex) = i.dir == DownIndex ? i : dual(i)
as_up(i::AbstractIndex) = i.dir == UpIndex ? i : dual(i)
