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
abstract type IxLoc end   # abstract type in Julia = Python ABC; no instances can be created; only exists to be subtyped

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
struct Upper <: IxLoc end   # empty struct = zero-size singleton. `<: IxLoc` makes Upper a subtype of IxLoc; this type TAG IS NOT INSTANTIATED IN USER CODE — it only exists as a type parameter to TIx{Upper}

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
struct Lower <: IxLoc end   # same as Upper: zero-size singleton, lives only as a type parameter

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
abstract type AbstractIx end   # root of the index hierarchy; analogous to Python `class AbstractIx(ABC): pass`

# These three functions are declared as abstract interface stubs — calling them on AbstractIx itself raises an error.
# In Python this would be: `@abstractmethod def dim(self): ...`
# The pattern: declare the interface here; concrete subtypes MUST provide methods, or they get the error at runtime.
dim(i::AbstractIx) = error("`dim` not implemented for $(typeof(i))")   # `$(typeof(i))` = Python f"{type(i).__name__}"; interpolated into the error string
label(i::AbstractIx) = error("`label` not implemented for $(typeof(i))")   # same pattern: interface stub; dispatch would reach this if a subtype forgot to implement `label`
which_space(i::AbstractIx) = error("`which_space` not implemented for $(typeof(i))")   # same pattern

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
struct TIx{L<:IxLoc} <: AbstractIx   # parametric struct: `{L<:IxLoc}` means L must be Upper or Lower; Python: `class TIx(AbstractIx, Generic[L])`; the `<: AbstractIx` makes TIx a subtype
    label::Symbol   # Symbol = Julia's interned string type; `:α`, `:σ_1`; unlike Python strings, Symbols have identity semantics (same symbol == same object)
    dim::Int        # local Hilbert space dimension; must be > 0 (checked in the constructor)

    function TIx{L}(label::Symbol, dim::Int) where {L<:IxLoc}   # inner constructor: the only way to create a TIx instance; `where {L<:IxLoc}` constrains the type parameter; Python: `def __init__(self, label, dim)` with validation
        dim > 0 || throw(ArgumentError("TIx dim must be a positive integer, got $dim."))   # `||` short-circuit: if `dim > 0` is false, throw; Python: `if not dim > 0: raise`; `$dim` = f-string interpolation
        new{L}(label, dim)   # `new{L}(...)` allocates and initialises the struct; must pass the type parameter explicitly in inner constructors; Python: `self.label = label` etc.
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
dim(ix::TIx) = ix.dim   # field access: `ix.dim` reads the stored Int; in Python: `ix.dim` or `ix._dim` with a property

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
label(ix::TIx) = ix.label   # field access; returns a Julia Symbol 

"""
    which_space(ix::TIx) -> Symbol

Return `:domain` for an upper (contravariant) index and `:codomain` for a lower
(covariant) index.

# Examples

```jldoctest
julia> which_space(TIx{Upper}(:α, 4))
:domain

julia> which_space(TIx{Lower}(:σ, 2))
:codomain
```
"""
which_space(::TIx{Upper}) = :domain    # dispatch on variance: upper index lives in the domain (input) space
which_space(::TIx{Lower}) = :codomain  # same pattern: separate method per variance tag; Julia compiles these as separate functions with zero overhead

# Extend Base.== and Base.hash so indices can be stored in dicts/sets and compared.
# Two indices are equal only if they have the same label, dimension, AND variance (Upper vs Lower).
# This is critical for contraction: we match Upper(label=:α, dim=2) with Lower(label=:α, dim=2),
# but not with Upper(label=:α, dim=2) or Upper(label=:β, dim=2). The hash enables fast index
# lookup when building contraction networks; dim is included so different dimensions don't
# collide, even if they happen to have the same label.
Base.:(==)(a::TIx{L}, b::TIx{L}) where {L<:IxLoc} = a.label == b.label && a.dim == b.dim   # Python: `__eq__`; `where {L<:IxLoc}` = both args must have the SAME variance tag L; `&&` = short-circuit AND
Base.:(==)(::TIx, ::TIx) = false   # catch-all for different variances (Upper vs Lower): they are never equal; `::TIx` without a type parameter matches any TIx; this method is less specific than the one above, so it only triggers when L differs
Base.hash(ix::TIx, h::UInt) = hash(ix.label, hash(ix.dim, hash(typeof(ix), h)))   # Python: `__hash__`; Julia's `hash(x, h)` = hash x seeded with h; `typeof(ix)` includes the variance parameter, so TIx{Upper}(:α,2) and TIx{Lower}(:α,2) hash differently; chained hashing like nested function calls

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
flip(ix::TIx{Upper}) = TIx{Lower}(ix.label, ix.dim)   # Upper → Lower: constructs new TIx with opposite variance, same label/dim; no data copy because TIx is purely symbolic
flip(ix::TIx{Lower}) = TIx{Upper}(ix.label, ix.dim)   # Lower → Upper: same pattern; two methods = involution

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
upper(label::Symbol, dim::Int) = TIx{Upper}(label, dim)   # convenience wrapper: shorter syntax than TIx{Upper}(:α, 4)

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
lower(label::Symbol, dim::Int) = TIx{Lower}(label, dim)   # convenience wrapper for Lower indices

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
uppers(pairs::Pair{Symbol,Int}...) = TIx{Upper}.(first.(pairs), last.(pairs))   # `pairs::Pair{Symbol,Int}...` = varargs of Pairs. `first.(pairs)` = broadcast `first` over pairs = extract all keys; `last.(pairs)` = extract all values; `TIx{Upper}.(...)` = broadcast constructor element-wise 

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
lowers(pairs::Pair{Symbol,Int}...) = TIx{Lower}.(first.(pairs), last.(pairs))   # same as `uppers` but for Lower variance

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
function uppers_range(base::Symbol, dim::Int, last::Int, start::Int=1)   # `start::Int=1` = keyword-less default argument. returns a Tuple (immutable, compile-time-sized)
    Tuple(TIx{Upper}(Symbol(base, :_, i), dim) for i in start:last)   # `Tuple(generator)` = collect a generator into an immutable Tuple 
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
function lowers_range(base::Symbol, dim::Int, last::Int, start::Int=1)   # same as uppers_range but creates Lower indices
    Tuple(TIx{Lower}(Symbol(base, :_, i), dim) for i in start:last)   # `Symbol(base, :_, i)` = symbol concatenation; e.g. Symbol(:χ, :_, 3) = :χ_3 (Python has no built-in Symbol type; closest is a string like "χ_3")
end
