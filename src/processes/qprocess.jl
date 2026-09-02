#=META
source:
  author: Bavithra
  coauthor: 
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs: coecke_kissinger_2016a, coecke_kissinger_2017
credits: 
=#

using TensorKit

# SECTION -  AbstractProcess: root of the process-theory hierarchy

"""
    AbstractProcess

Root of the Qritical.jl process hierarchy, following Coecke & Kissinger's diagrammatic
process-theory picture (*Picturing Quantum Processes*, §3.3-3.4): a **process** is a box with
some number of input wires and some number of output wires; wiring a [`TIx`](@ref)/[`MulTIx`](@ref)
onto each wire is exactly what distinguishes a bare `TensorKit.TensorMap` (numbers in a basis)
from a process (a physically meaningful box whose legs carry system-type information).

Every concrete subtype wraps a `TensorKit.AbstractTensorMap` and exposes it, along with its
output/input legs, through the common accessor interface [`tensor`](@ref), [`outputs`](@ref), and
[`inputs`](@ref) - this lets generic code (today [`is_state`](@ref)/[`is_effect`](@ref)/
[`equal_up_to_scalar`](@ref); later sequential/parallel composition) be written once against
`AbstractProcess` rather than once per concrete type.

Concrete subtypes: [`QProcess`](@ref) (the general case), [`State`](@ref) (no inputs),
[`Effect`](@ref) (no outputs), [`Scalar`](@ref) (neither).

!!! note "Linear maps, not quantum maps"

    This models Coecke & Kissinger's process theory of **linear maps** (states are kets, effects
    are bras/functionals, numbers are amplitudes) - exactly what a single `TensorKit.TensorMap`
    represents. The *quantum maps* picture (states as density operators, effects as positive
    functionals `ρ ↦ tr(Aρ)`) is a distinct, richer process theory built on doubled/CP-map
    structure (their §3.4.1, Example* 3.37) and is not modelled here.
"""
abstract type AbstractProcess end

# SECTION -  Wiring utilities: TensorMap -> TIx legs

"""
    codomain_legs(t::TensorKit.AbstractTensorMap; roles=PhysicalLeg()) -> Tuple{Vararg{TIx}}

Build one [`TIx`](@ref) per leg of `t`'s codomain (its output/"ket" side), reading each leg's
underlying `TensorKit.ElementarySpace` directly off `TensorKit.codomain(t)`.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:tensor]))

# Keywords

$(Glossaries.Keyword{@__MODULE__}()([:roles]))

# Examples

```jldoctest
julia> using TensorKit;

julia> V = ComplexSpace(2);

julia> t = zeros(ComplexF64, V ⊗ V ← one(V));

julia> codomain_legs(t) == (TIx(V), TIx(V))
true

julia> codomain_legs(t; roles=VirtualLeg()) == (TIx(V, VirtualLeg()), TIx(V, VirtualLeg()))
true
```
"""
function codomain_legs(
    t::TensorKit.AbstractTensorMap;
    roles::Union{LegRole,Tuple{Vararg{LegRole}}}=PhysicalLeg(),
)
    return _wire_legs(TensorKit.codomain(t), roles, "codomain")
end

"""
    domain_legs(t::TensorKit.AbstractTensorMap; roles=PhysicalLeg()) -> Tuple{Vararg{TIx}}

Build one [`TIx`](@ref) per leg of `t`'s domain (its input/"bra" side), reading each leg's
underlying `TensorKit.ElementarySpace` directly off `TensorKit.domain(t)`.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:tensor]))

# Keywords

$(Glossaries.Keyword{@__MODULE__}()([:roles]))

# Examples

```jldoctest
julia> using TensorKit;

julia> V = ComplexSpace(2);

julia> t = zeros(ComplexF64, one(V) ← V);

julia> domain_legs(t) == (TIx(V),)
true
```
"""
function domain_legs(
    t::TensorKit.AbstractTensorMap;
    roles::Union{LegRole,Tuple{Vararg{LegRole}}}=PhysicalLeg(),
)
    return _wire_legs(TensorKit.domain(t), roles, "domain")
end

# Shared implementation: build a TIx per leg of a ProductSpace, broadcasting or matching `roles`.
function _wire_legs(
    spaces::TensorKit.ProductSpace,
    roles::Union{LegRole,Tuple{Vararg{LegRole}}},
    side::String,
)
    n = length(spaces)
    role_tuple = roles isa LegRole ? ntuple(_ -> roles, n) : roles
    length(role_tuple) == n || throw(
        ArgumentError(
            "Expected $n leg role(s) for the $side (got $(length(role_tuple))); pass a single LegRole to broadcast to every leg, or a Tuple{Vararg{LegRole}} matching leg count.",
        ),
    )
    return ntuple(i -> TIx(spaces[i], role_tuple[i]), n)
end

# SECTION -  QProcess: the general case, legs on both sides

"""
    QProcess{T<:TensorKit.AbstractTensorMap} <: AbstractProcess

A process with legs on both sides: some number of output legs (`t`'s codomain) and some number of
input legs (`t`'s domain), each wired up as a [`TIx`](@ref) via [`codomain_legs`](@ref)/
[`domain_legs`](@ref). This is the general box in Coecke & Kissinger's diagrammatic language
(§3.3-3.4); [`State`](@ref), [`Effect`](@ref), and [`Scalar`](@ref) are its special cases with one
or both sides forced empty.

# Fields

$(Glossaries.Field{@__MODULE__}()([:tensor]))

  - `outputs :: Tuple{Vararg{AbstractIx}}` - the codomain legs, one per leg of `tensor`'s codomain
  - `inputs  :: Tuple{Vararg{AbstractIx}}` - the domain legs, one per leg of `tensor`'s domain

The inner constructor validates that `outputs`/`inputs` have the right length
(`TensorKit.numout(tensor)`/`TensorKit.numin(tensor)`) and that each leg's `space` matches
`tensor`'s actual codomain/domain space at that position, throwing `DimensionMismatch`/
`ArgumentError` otherwise.

# Examples

```jldoctest
julia> using TensorKit;

julia> V = ComplexSpace(2);

julia> t = zeros(ComplexF64, V ← V);

julia> p = QProcess(t);

julia> outputs(p) == (TIx(V),)
true

julia> inputs(p) == (TIx(V),)
true
```
"""
struct QProcess{T<:TensorKit.AbstractTensorMap} <: AbstractProcess
    tensor::T
    outputs::Tuple{Vararg{AbstractIx}}
    inputs::Tuple{Vararg{AbstractIx}}

    function QProcess(
        tensor::T, outputs::Tuple{Vararg{AbstractIx}}, inputs::Tuple{Vararg{AbstractIx}}
    ) where {T<:TensorKit.AbstractTensorMap}
        length(outputs) == TensorKit.numout(tensor) || throw(
            DimensionMismatch(
                "Expected $(TensorKit.numout(tensor)) output leg(s), got $(length(outputs)).",
            ),
        )
        length(inputs) == TensorKit.numin(tensor) || throw(
            DimensionMismatch(
                "Expected $(TensorKit.numin(tensor)) input leg(s), got $(length(inputs)).",
            ),
        )
        codomain_spaces = TensorKit.codomain(tensor)
        domain_spaces = TensorKit.domain(tensor)
        all(space(outputs[i]) == codomain_spaces[i] for i in eachindex(outputs)) || throw(
            ArgumentError(
                "An output leg's space does not match tensor's codomain at that position.",
            ),
        )
        all(space(inputs[i]) == domain_spaces[i] for i in eachindex(inputs)) || throw(
            ArgumentError(
                "An input leg's space does not match tensor's domain at that position."
            ),
        )
        return new{T}(tensor, outputs, inputs)
    end
end

"""
    QProcess(t::TensorKit.AbstractTensorMap; output_roles=PhysicalLeg(), input_roles=PhysicalLeg())

Build a [`QProcess`](@ref) from `t`, deriving its output/input legs automatically via
[`codomain_legs`](@ref)/[`domain_legs`](@ref).

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:tensor]))

# Keywords

  - `output_roles` - $(Glossaries.Plain{@__MODULE__}(:description)(:roles)); applied to the output legs.
  - `input_roles`  - $(Glossaries.Plain{@__MODULE__}(:description)(:roles)); applied to the input legs.
"""
function QProcess(
    t::TensorKit.AbstractTensorMap;
    output_roles::Union{LegRole,Tuple{Vararg{LegRole}}}=PhysicalLeg(),
    input_roles::Union{LegRole,Tuple{Vararg{LegRole}}}=PhysicalLeg(),
)
    return QProcess(
        t, codomain_legs(t; roles=output_roles), domain_legs(t; roles=input_roles)
    )
end

# SECTION -  State: a process with no inputs (a preparation, a ket)

"""
    State{T<:TensorKit.AbstractTensorMap{<:Any,<:Any,<:Any,0}} <: AbstractProcess

A process with **no inputs** - a preparation procedure, Coecke & Kissinger's ket (§3.4.1). The
trivial domain (`TensorKit.numin(tensor) == 0`) is pinned directly in the type via TensorKit's own
`N₂` type parameter on `AbstractTensorMap{T,S,N₁,N₂}`, so `State` only ever stores its output legs.

Constructing a `State{T}` directly with a wrong-shaped tensor (nonzero domain) fails via this type
parameter, at the type-system level, not a runtime check; the friendly outer constructor
[`State(t::TensorKit.AbstractTensorMap; roles)`](@ref) checks first and raises a clear
`ArgumentError` instead.

!!! note "Uniform-rank boundary tensors"

    An MPS's boundary site tensor is often stored with a nominal dimension-1 bond leg rather than
    genuinely having one fewer leg (so every site tensor in the chain has the same rank). Such a
    tensor does not yet satisfy `numin(tensor) == 0` and so is not directly a `State`.
    `TensorKit.removeunit` strips that placeholder leg down to the true zero-leg shape a `State`
    requires; `TensorKit.insertleftunit`/`insertrightunit` do the reverse, for putting a `State`
    back into uniform-rank storage inside a network.

# Fields

$(Glossaries.Field{@__MODULE__}()([:tensor]))

  - `outputs :: Tuple{Vararg{AbstractIx}}` - the codomain legs, one per leg of `tensor`'s codomain
"""
struct State{T<:TensorKit.AbstractTensorMap{<:Any,<:Any,<:Any,0}} <: AbstractProcess
    tensor::T
    outputs::Tuple{Vararg{AbstractIx}}
end

"""
    State(t::TensorKit.AbstractTensorMap; roles=PhysicalLeg())

Build a [`State`](@ref) from `t`, deriving its output legs automatically via
[`codomain_legs`](@ref). Throws `ArgumentError` if `t` does not have a trivial domain
(`TensorKit.numin(t) != 0`).

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:tensor]))

# Keywords

$(Glossaries.Keyword{@__MODULE__}()([:roles]))

# Examples

```jldoctest
julia> using TensorKit;

julia> V = ComplexSpace(2);

julia> ψ = zeros(ComplexF64, V ← one(V));

julia> ψ[1] = 1;

julia> s = State(ψ);

julia> is_state(s)
true
```
"""
function State(
    t::TensorKit.AbstractTensorMap;
    roles::Union{LegRole,Tuple{Vararg{LegRole}}}=PhysicalLeg(),
)
    TensorKit.numin(t) == 0 || throw(
        ArgumentError(
            "A State must have a trivial domain (zero inputs); got numin(t) = $(TensorKit.numin(t)).",
        ),
    )
    return State(t, codomain_legs(t; roles=roles))
end

# SECTION -  Effect: a process with no outputs (a test, a bra)

"""
    Effect{T<:TensorKit.AbstractTensorMap{<:Any,<:Any,0,<:Any}} <: AbstractProcess

A process with **no outputs** - a test, Coecke & Kissinger's bra (§3.4.1). The trivial codomain
(`TensorKit.numout(tensor) == 0`) is pinned directly in the type via TensorKit's own `N₁` type
parameter, so `Effect` only ever stores its input legs. Mirror image of [`State`](@ref): the same
`TensorKit.removeunit`/`insertleftunit`/`insertrightunit` connection to uniform-rank boundary
tensors applies here too.

# Fields

$(Glossaries.Field{@__MODULE__}()([:tensor]))

  - `inputs :: Tuple{Vararg{AbstractIx}}` - the domain legs, one per leg of `tensor`'s domain
"""
struct Effect{T<:TensorKit.AbstractTensorMap{<:Any,<:Any,0,<:Any}} <: AbstractProcess
    tensor::T
    inputs::Tuple{Vararg{AbstractIx}}
end

"""
    Effect(t::TensorKit.AbstractTensorMap; roles=PhysicalLeg())

Build an [`Effect`](@ref) from `t`, deriving its input legs automatically via
[`domain_legs`](@ref). Throws `ArgumentError` if `t` does not have a trivial codomain
(`TensorKit.numout(t) != 0`).

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:tensor]))

# Keywords

$(Glossaries.Keyword{@__MODULE__}()([:roles]))

# Examples

```jldoctest
julia> using TensorKit;

julia> V = ComplexSpace(2);

julia> π = zeros(ComplexF64, one(V) ← V);

julia> π[1] = 1;

julia> e = Effect(π);

julia> is_effect(e)
true
```
"""
function Effect(
    t::TensorKit.AbstractTensorMap;
    roles::Union{LegRole,Tuple{Vararg{LegRole}}}=PhysicalLeg(),
)
    TensorKit.numout(t) == 0 || throw(
        ArgumentError(
            "An Effect must have a trivial codomain (zero outputs); got numout(t) = $(TensorKit.numout(t)).",
        ),
    )
    return Effect(t, domain_legs(t; roles=roles))
end

# SECTION -  Scalar: a process with neither inputs nor outputs (a number)

"""
    Scalar{T<:TensorKit.AbstractTensorMap{<:Any,<:Any,0,0}} <: AbstractProcess

A process with **neither inputs nor outputs** - Coecke & Kissinger's number (§3.4.1): "processes
without inputs or outputs are just a set of things that can be multiplied." What you get when a
[`State`](@ref) and an [`Effect`](@ref) are composed (e.g. `⟨ψ|ψ⟩`). Both `N₁` and `N₂` are pinned
to `0` directly in the type, so a `Scalar` reduces, up to basis, to a single genuine number
regardless of whether its legs would have carried symmetry sectors - with zero legs there are no
sectors left to distribute over, only the trivial one.

!!! note "Tracking a normalization factor without collapsing to a number yet"

    A sweep that needs to track gauge/normalization freedom (see [`equal_up_to_scalar`](@ref)) can
    carry the leftover factor as a dimension-1 leg via `TensorKit.insertleftunit`, deferring the
    actual `Scalar`/[`value`](@ref) extraction (`TensorKit.removeunit` then `Scalar(...)`) until
    the number is actually needed.

# Fields

$(Glossaries.Field{@__MODULE__}()([:tensor]))

# Examples

```jldoctest
julia> using TensorKit;

julia> V = ComplexSpace(2);

julia> t = TensorMap(fill(3.0 + 0im), one(V), one(V));

julia> value(Scalar(t))
3.0 + 0.0im
```
"""
struct Scalar{T<:TensorKit.AbstractTensorMap{<:Any,<:Any,0,0}} <: AbstractProcess
    tensor::T
end

"""
    value(q::Scalar) -> Number

Pull the genuine number out of `q`, via `TensorKit.scalar` (which already exists precisely for a
`TensorMap` with trivial domain and codomain).
"""
value(q::Scalar) = TensorKit.scalar(q.tensor)

# SECTION -  Uniform accessors across the AbstractProcess family

"""
    tensor(p::AbstractProcess) -> TensorKit.AbstractTensorMap

Returns $(Glossaries.Plain{@__MODULE__}(:description)(:tensor)).
"""
tensor(p::QProcess) = p.tensor
tensor(p::State) = p.tensor
tensor(p::Effect) = p.tensor
tensor(p::Scalar) = p.tensor

"""
    outputs(p::AbstractProcess) -> Tuple{Vararg{AbstractIx}}

The output (codomain) legs of `p`; `()` for an [`Effect`](@ref) or [`Scalar`](@ref), which have
none.
"""
outputs(p::QProcess) = p.outputs
outputs(p::State) = p.outputs
outputs(::Effect) = ()
outputs(::Scalar) = ()

"""
    inputs(p::AbstractProcess) -> Tuple{Vararg{AbstractIx}}

The input (domain) legs of `p`; `()` for a [`State`](@ref) or [`Scalar`](@ref), which have none.
"""
inputs(p::QProcess) = p.inputs
inputs(::State) = ()
inputs(p::Effect) = p.inputs
inputs(::Scalar) = ()

# SECTION -  Predicates

"""
    is_state(p::AbstractProcess) -> Bool

`true` iff `p` has no input legs - a preparation/ket, per Coecke & Kissinger §3.4.1.
"""
is_state(p::AbstractProcess) = isempty(inputs(p))

"""
    is_effect(p::AbstractProcess) -> Bool

`true` iff `p` has no output legs - a test/bra, per Coecke & Kissinger §3.4.1.
"""
is_effect(p::AbstractProcess) = isempty(outputs(p))

# SECTION -  equal_up_to_scalar: Coecke & Kissinger's Definition 3.39

"""
    equal_up_to_scalar(p::AbstractProcess, q::AbstractProcess; atol=0, rtol=...) -> Bool

Coecke & Kissinger's Definition 3.39: two processes `f ≈ g` iff there exist **nonzero** numbers
`λ, μ` with `λf = μg`. Useful for tracking gauge/normalization freedom in a tensor network
calculation - e.g. checking that two differently-normalized copies of the same MPS tensor are
"the same tensor" up to scale.

For nonzero tensors, this is equivalent to `tensor(p)` and `tensor(q)` being parallel as vectors,
which is checked via the Cauchy-Schwarz equality case `|⟨f,g⟩| ≈ ‖f‖‖g‖`, using
`TensorKit.dot`/`TensorKit.norm` directly (both work generically on any `AbstractTensorMap` shape,
trivial or symmetric). Mismatched leg spaces return `false` rather than throwing. The all-zero
case (`p` and `q` both the zero process) returns `true`; one zero and one nonzero returns `false`,
since no nonzero `λ, μ` can relate them.
"""
function equal_up_to_scalar(
    p::AbstractProcess,
    q::AbstractProcess;
    atol::Real=0,
    rtol::Real=Base.rtoldefault(Float64),
)
    tp, tq = tensor(p), tensor(q)
    TensorKit.space(tp) == TensorKit.space(tq) || return false
    np, nq = TensorKit.norm(tp), TensorKit.norm(tq)
    if iszero(np) && iszero(nq)
        return true
    elseif iszero(np) || iszero(nq)
        return false
    end
    return isapprox(abs(TensorKit.dot(tp, tq)), np * nq; atol=atol, rtol=rtol)
end
