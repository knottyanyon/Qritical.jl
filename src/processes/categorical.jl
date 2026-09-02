#=META
source:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
docstrings:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
refs: coecke_kissinger_2016a, coecke_kissinger_2017
credits: N/A
=#

using TensorKit

# SECTION -  _wrap_process: pick the right AbstractProcess subtype from a bare tensor

"""
    _wrap_process(t, outputs, inputs) -> AbstractProcess

Internal helper shared by [`Base.:∘`](@ref) and [`TensorKit.:⊗`](@ref): given a freshly computed
`TensorKit.AbstractTensorMap` `t` and the `outputs`/`inputs` leg tuples it should carry, pick the
right concrete [`AbstractProcess`](@ref) subtype from `t`'s actual `numout`/`numin` - `Scalar` for
`(0, 0)`, `State` for `(n, 0)`, `Effect` for `(0, n)`, `QProcess` otherwise. This is exactly the
same narrowing [`QProcess`](@ref)'s own inner constructor validates against, just applied once
here so `∘`/`⊗` don't each duplicate the dispatch.
"""
function _wrap_process(
    t::TensorKit.AbstractTensorMap,
    outputs::Tuple{Vararg{AbstractIx}},
    inputs::Tuple{Vararg{AbstractIx}},
)
    if TensorKit.numout(t) == 0 && TensorKit.numin(t) == 0
        return Scalar(t)
    elseif TensorKit.numin(t) == 0
        return State(t, outputs)
    elseif TensorKit.numout(t) == 0
        return Effect(t, inputs)
    else
        return QProcess(t, outputs, inputs)
    end
end

# SECTION -  adjoint (dagger): Coecke & Kissinger §3.4.4, Definition 4.97

"""
    adjoint(p::AbstractProcess) -> AbstractProcess
    p'

The dagger of `p` (Coecke & Kissinger's `†`, Definition 4.97): reverses `p`'s wires, turning a
[`State`](@ref) (ket) into an [`Effect`](@ref) (bra) and vice versa, via
`TensorKit.adjoint(tensor(p))`. `p`'s own leg tuples are reused, swapped (`p`'s old `inputs`
become the new `outputs` and vice versa) rather than re-derived from scratch via
[`codomain_legs`](@ref)/[`domain_legs`](@ref) - this preserves each leg's exact identity
(`LegRole` included) across the flip, and is valid precisely because every space this codebase
constructs (`ComplexSpace`, `Z2`-graded spaces, ...) has `TensorKit.InnerProductStyle === EuclideanInnerProduct()` (self-dual: `dual(V) === V`), so the adjointed tensor's new
codomain/domain spaces match the reused legs' spaces exactly. This is also what makes `adjoint`
an honest involution (`p'' == p`, `==`-equal legs and all) as Definition 4.97 requires of a
dagger functor.

Exported under the alias [`dagger`](@ref) for readers using Coecke & Kissinger's own vocabulary.

# Examples

```jldoctest
julia> using TensorKit;

julia> V = ComplexSpace(2);

julia> ψ = zeros(ComplexF64, V ← one(V));

julia> ψ[1] = 1;

julia> s = State(ψ);

julia> is_effect(s')
true
```
"""
Base.adjoint(p::QProcess) = QProcess(TensorKit.adjoint(tensor(p)), inputs(p), outputs(p))
Base.adjoint(p::State) = Effect(TensorKit.adjoint(tensor(p)), outputs(p))
Base.adjoint(p::Effect) = State(TensorKit.adjoint(tensor(p)), inputs(p))
Base.adjoint(p::Scalar) = Scalar(TensorKit.adjoint(tensor(p)))

"""
    dagger(p::AbstractProcess) -> AbstractProcess

Alias for [`Base.adjoint`](@ref) (`p'`), spelled out for readers using Coecke & Kissinger's `†`
vocabulary directly.
"""
const dagger = adjoint

# SECTION -  Sequential composition: Coecke & Kissinger §3.2.2

"""
    ∘(g::AbstractProcess, f::AbstractProcess) -> AbstractProcess

Sequential composition (Coecke & Kissinger §3.2.2): `f` happens first, then `g`, connecting `f`'s
outputs to `g`'s inputs. Requires `inputs(g) == outputs(f)` as an *ordered* tuple equality (no leg
reordering/swapping is performed - see the module note on scope); on a mismatch, throws
`DimensionMismatch`/`ArgumentError` naming the offending position, mirroring
[`QProcess`](@ref)'s own inner-constructor error style.

Dispatches to the right result type via [`_wrap_process`](@ref) - e.g. `Effect ∘ State` yields a
[`Scalar`](@ref) (Coecke & Kissinger's `⟨ψ|ψ⟩`, §3.4.1's Born-rule pattern).

# Examples

```jldoctest
julia> using TensorKit;

julia> V = ComplexSpace(2);

julia> ψ = zeros(ComplexF64, V ← one(V));

julia> ψ[1] = 1;

julia> s = State(ψ);

julia> value(s' ∘ s)
1.0 + 0.0im
```
"""
function Base.:∘(g::AbstractProcess, f::AbstractProcess)
    fo, gi = outputs(f), inputs(g)
    length(gi) == length(fo) || throw(
        DimensionMismatch(
            "Cannot compose: g has $(length(gi)) input leg(s), f has $(length(fo)) output leg(s).",
        ),
    )
    for (k, (a, b)) in enumerate(zip(gi, fo))
        a == b || throw(
            ArgumentError(
                "Cannot compose at leg $k: g's input leg $a does not match f's output leg $b.",
            ),
        )
    end
    return _wrap_process(tensor(g) * tensor(f), outputs(g), inputs(f))
end

# SECTION -  Parallel composition: Coecke & Kissinger §3.2.1

"""
    ⊗(p::AbstractProcess, q::AbstractProcess) -> AbstractProcess

Parallel composition (Coecke & Kissinger §3.2.1): `p` and `q` happen independently, side by side.
Implemented via `TensorKit.⊗` on the underlying tensors, with `outputs`/`inputs` concatenated in
order (`p`'s legs first, then `q`'s). Dispatches to the right result type via
[`_wrap_process`](@ref).

# Examples

```jldoctest
julia> using TensorKit;

julia> V = ComplexSpace(2);

julia> ψ = zeros(ComplexF64, V ← one(V));
       ψ[1] = 1;

julia> φ = zeros(ComplexF64, V ← one(V));
       φ[2] = 1;

julia> length(outputs(State(ψ) ⊗ State(φ)))
2
```
"""
function TensorKit.:⊗(p::AbstractProcess, q::AbstractProcess)
    return _wrap_process(
        tensor(p) ⊗ tensor(q), (outputs(p)..., outputs(q)...), (inputs(p)..., inputs(q)...)
    )
end

# SECTION -  identity_process: the resolution of the identity (Coecke & Kissinger §3.1.3)

"""
    identity_process(ix::AbstractIx) -> QProcess

The identity process on `ix` (Coecke & Kissinger's `1_A`, §3.1.3, Definition 3.42's `δ^B_A`):
`TensorKit.id(space(ix))`, with one output leg and one input leg both equal to `ix`. This is
literally Coecke & Kissinger's Theorem 5.31 "resolution of the identity", evaluated on the
orthonormal basis `TIx` fixes for `ix`'s space - not an analogy to it.

# Examples

```jldoctest
julia> using TensorKit;

julia> ix = TIx(3);

julia> p = identity_process(ix);

julia> outputs(p) == (ix,) && inputs(p) == (ix,)
true
```
"""
function identity_process(ix::AbstractIx)
    return QProcess(TensorKit.id(space(ix)), (ix,), (ix,))
end

# SECTION -  is_isometry / is_unitary: Coecke & Kissinger Propositions 5.37/5.38

"""
    is_isometry(p::AbstractProcess; isapprox_kwargs...) -> Bool

`true` iff `p` is an isometry: `dagger(p) ∘ p ≈ 1` on `p`'s input system (Coecke & Kissinger
Proposition 5.37) - the **one-sided** equation, never the two-sided unitary one (see
[`is_unitary`](@ref)). Delegates to `TensorKit.isisometric(tensor(p); side=:left)` (re-exported
from `MatrixAlgebraKit.jl`'s `common/matrixproperties.jl`), which already implements exactly this
equation - `A'*A ≈ I` - generically over any `AbstractTensorMap`, with proper `atol`/`rtol`
handling built in; this is what makes a canonicalized MPS site tensor's isometry condition a
real, checkable library call rather than a bespoke `norm(A'*A - I)` calculation.

# Examples

```jldoctest
julia> using TensorKit;

julia> V = ComplexSpace(2);

julia> u, _, _ = svd_compact(randn(ComplexF64, V ← V));

julia> is_isometry(QProcess(u))
true
```
"""
function is_isometry(p::AbstractProcess; isapprox_kwargs...)
    return TensorKit.isisometric(tensor(p); side=:left, isapprox_kwargs...)
end

"""
    is_unitary(p::AbstractProcess; isapprox_kwargs...) -> Bool

`true` iff `p` is unitary (Coecke & Kissinger Proposition 5.38's two-sided equation, `dagger(p) ∘ p ≈ 1` **and** `p ∘ dagger(p) ≈ 1`). Delegates to `TensorKit.isunitary(tensor(p))` (re-exported
from `MatrixAlgebraKit.jl`), which checks both sides directly rather than calling
[`is_isometry`](@ref) twice.
"""
function is_unitary(p::AbstractProcess; isapprox_kwargs...)
    return TensorKit.isunitary(tensor(p); isapprox_kwargs...)
end
