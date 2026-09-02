#=META
source:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
docstrings:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
refs:
credits: mpskit-validation (internal validation package) - decompositions.jl
=#

using ..SimStudy:
    SimStudy, RecordingTrait, Active, record!, finalize!, AbstractErrorAccumulator

# The type-tag shape (ProductFormula{Order}/Trotterization{PF}/OperatorSplitting{F}) mirrors
# mpskit-validation/decompositions.jl's bare stub declarations, but everything below them -
# `sequence`, `local_error_bound`, `TrotterErrorAccumulator` - is new: that file declared the
# names and never implemented any coefficient-sequencing or error-tracking logic anywhere.
# `decompositions.jl`:26-29 in this package flags this exact piece of work as belonging in "a
# future TEBD subroutine file, not here" - this is that file. Deliberately has zero dependency on
# `QProcess`/`TensorMap`/exponentiation: `terms` stays fully opaque throughout, so this stays
# usable from anywhere a caller has an ordered list of splittable pieces, not only a TEBD sweep -
# e.g. a future `Propagator` operator type that wraps `sequence`/`TrotterErrorAccumulator` to
# actually exponentiate and apply terms.

# SECTION -  ApproximateDecomposition / ProductFormula / Trotterization / OperatorSplitting - type hierarchy

"""
    ApproximateDecomposition

Abstract root for decomposition strategies that only approximate an exact operation, as opposed
to [`ExactDecomposition`](@ref)'s exact matrix factorizations. [`Trotterization`](@ref) and
[`OperatorSplitting`](@ref) both approximate an operator exponential/generator split rather than
exactly factorizing a tensor, so they sit here instead.
"""
abstract type ApproximateDecomposition end

"""
    ProductFormula{Order}

Abstract root of the product-formula order tags - `Order` is a plain `Int` type parameter (the
same idiom [`TensorTrain`](@ref)'s site-arity parameter `P` uses), not a trait type. Dispatch
happens on the concrete singleton subtypes below, never by inspecting `Order` at runtime.

Concrete subtypes: [`LieTrotter`](@ref) (`Order=1`), [`SuzukiTrotter`](@ref) (`Order=2`),
[`Suzuki4th`](@ref) (`Order=4`).
"""
abstract type ProductFormula{Order} end

"""
1st-order (Lie-Trotter) product formula: naive term-by-term sequencing, local error `O(dt²)`. See
[`ProductFormula`](@ref).
"""
struct LieTrotter <: ProductFormula{1} end

"""
2nd-order (Suzuki-Trotter / Strang splitting) product formula: half-step forward through the
terms, full step on the last, half-step back. Local error `O(dt³)`. See [`ProductFormula`](@ref).
"""
struct SuzukiTrotter <: ProductFormula{2} end

"""
4th-order (Suzuki's recursive) product formula, built from three nested [`SuzukiTrotter`](@ref)
calls via the `s = 1/(4 - 4^(1/3))` coefficient combination. Local error `O(dt⁵)`. See
[`ProductFormula`](@ref).
"""
struct Suzuki4th <: ProductFormula{4} end

"""
    Trotterization{PF<:ProductFormula}

Abstract extension point for a future TEBD driver to specialize a full Trotterization strategy
(e.g. how sub-steps get applied/truncated) on top of a chosen [`ProductFormula`](@ref). No
concrete subtypes or methods exist yet - this file only implements [`sequence`](@ref), dispatched
directly on the `ProductFormula` singletons, not through this layer. Mirrors `InfiniteSupport`'s
role in [`BoundarySupport`](@ref): declared now, unused until a concrete need arrives.
"""
abstract type Trotterization{PF<:ProductFormula} <: ApproximateDecomposition end

"""
    OperatorSplitting{F<:Trotterization}

Abstract extension point one level above [`Trotterization`](@ref), for a future generalization
beyond product-formula Trotterization (e.g. other operator-splitting schemes). No concrete
subtypes or methods exist yet - see [`Trotterization`](@ref).
"""
abstract type OperatorSplitting{F<:Trotterization} <: ApproximateDecomposition end

# SECTION -  sequence - the (term, coefficient) recipe for one Trotter step

"""
    sequence(::LieTrotter, terms, dt) -> Vector{Tuple{Any,Float64}}
    sequence(::SuzukiTrotter, terms, dt) -> Vector{Tuple{Any,Float64}}
    sequence(::Suzuki4th, terms, dt) -> Vector{Tuple{Any,Float64}}

The ordered `(term, coefficient)` sub-step recipe for **one** Trotter step at the order tagged by
the first (singleton) argument: `coefficient` multiplies `dt` in that sub-step's exponent, i.e.
the recipe says "apply `exp(coefficient * dt * term)` here" without this function ever computing
that exponential or inspecting `term` itself - `terms` stays fully opaque throughout.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:pf_terms, :pf_dt]))

Repeating the returned recipe `num_steps` times, for a full time evolution, is left to the caller

  - `sequence` always returns a single step's recipe, never `num_steps` copies of it.
"""
function sequence(::LieTrotter, terms::AbstractVector, dt)
    return [(term, 1.0) for term in terms]
end

function sequence(::SuzukiTrotter, terms::AbstractVector, dt)
    isempty(terms) && return Tuple{Any,Float64}[]
    forward = [(term, 0.5) for term in terms[1:(end - 1)]]
    last_step = (terms[end], 1.0)
    backward = [(term, 0.5) for term in terms[(end - 1):-1:1]]
    return vcat(forward, [last_step], backward)
end

function _rescale_coefficients(sub_sequence, factor)
    return [(term, c * factor) for (term, c) in sub_sequence]
end

"""
    sequence(::Suzuki4th, terms, dt) -> Vector{Tuple{Any,Float64}}

Recursive construction: `SuzukiTrotter`'s own dimensionless splitting weights (`0.5`/`1.0`/`0.5`)
never depend on the numeric value of its `dt` argument, so composing `S2(s*dt) S2(s*dt) S2((1-4s)*dt) S2(s*dt) S2(s*dt)` is done here by calling `sequence(SuzukiTrotter(), terms, dt)`
once per sub-formula and rescaling *its returned coefficients* by `s`/`(1-4s)` afterwards, rather
than by passing a rescaled timestep into each nested call (which, since those coefficients don't
read `dt` at all, would have no effect on what's returned). This keeps every returned tuple's
`coefficient` field correctly meaning "multiplies the *outer* `dt` passed to this call" while
still genuinely reusing [`SuzukiTrotter`](@ref)'s own `sequence` method for the nested structure,
rather than hand-flattening the 4th-order formula.
"""
function sequence(::Suzuki4th, terms::AbstractVector, dt)
    s = 1 / (4 - 4^(1 / 3))
    return vcat(
        _rescale_coefficients(sequence(SuzukiTrotter(), terms, dt), s),
        _rescale_coefficients(sequence(SuzukiTrotter(), terms, dt), s),
        _rescale_coefficients(sequence(SuzukiTrotter(), terms, dt), 1 - 4s),
        _rescale_coefficients(sequence(SuzukiTrotter(), terms, dt), s),
        _rescale_coefficients(sequence(SuzukiTrotter(), terms, dt), s),
    )
end

# SECTION -  TrotterErrorAccumulator - a priori Trotter/splitting error tracking

"""
    local_error_bound(pf::ProductFormula, terms, dt, norm) -> Float64

A priori bound on one step's local Trotter error at the order tagged by `pf`, using the
submultiplicative commutator bound (`‖[A,B]‖ ≤ 2‖A‖‖B‖`) rather than computing any actual
commutator or exponential: with `N = sum(norm, terms)` the aggregate operator norm of the terms
being split, the bound scales as `C(order) * (N*dt)^(order+1)` - the standard leading-order
scaling (`O(dt²)` for [`LieTrotter`](@ref), `O(dt³)` for [`SuzukiTrotter`](@ref), `O(dt⁵)` for
[`Suzuki4th`](@ref)). `C(order)` is a small fixed, documented, *a priori* constant (`C(1)=1`,
`C(2)=1/12`, `C(4)=1`) - not a tuned/fitted value, so `C(4)` is left at the same unit prefactor as
`C(1)` rather than asserting a tighter derived constant this file hasn't actually derived.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:pf_terms, :pf_dt, :pf_norm]))
"""
function local_error_bound(::LieTrotter, terms::AbstractVector, dt, norm)
    N = sum(norm, terms)
    return (N * dt)^2
end
function local_error_bound(::SuzukiTrotter, terms::AbstractVector, dt, norm)
    N = sum(norm, terms)
    return (N * dt)^3 / 12
end
function local_error_bound(::Suzuki4th, terms::AbstractVector, dt, norm)
    N = sum(norm, terms)
    return (N * dt)^5
end

"""
    TrotterErrorAccumulator()

Tracks the running Trotter/splitting error across multiple steps via **plain summation** of each
step's [`local_error_bound`](@ref) - a deliberate divergence from
`QuadratureTruncationErrorAccumulator`'s `hypot` combination: Trotter error is a systematic bias
that adds linearly across steps (`O(num_steps * dt^(order+1)) = O(T*dt^order)` for total evolution
time `T`), not an independent-contributions quantity the way separate SVD truncations are (where
2-norms add in quadrature). Built on the same `AbstractErrorAccumulator`/`RecordingTrait`
machinery as the truncation accumulator, so a future TEBD sweep can hold both side by side and
combine them the same way onto a chain's total error.

# Fields

  - `total   :: Float64`         - running summed total
  - `history :: Vector{Float64}` - per-step raw `local_error` values, in the order recorded
"""
mutable struct TrotterErrorAccumulator <: AbstractErrorAccumulator
    total::Float64
    history::Vector{Float64}
end
TrotterErrorAccumulator() = TrotterErrorAccumulator(0.0, Float64[])

function record!(::Active, acc::TrotterErrorAccumulator, ctx::NamedTuple)
    acc.total += ctx.local_error
    push!(acc.history, ctx.local_error)
    return acc.total
end
finalize!(::Active, acc::TrotterErrorAccumulator) = acc.total

"""
    accumulate_trotter_error!(acc, pf::ProductFormula, terms, dt, num_steps, norm) -> Float64

Convenience wrapper that calls [`local_error_bound`](@ref) and `record!` once per step for
`num_steps` identical steps, then returns `finalize!(acc)`.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:pf_terms, :pf_dt, :pf_norm]))
"""
function accumulate_trotter_error!(
    acc::AbstractErrorAccumulator,
    pf::ProductFormula,
    terms::AbstractVector,
    dt,
    num_steps::Int,
    norm,
)
    for step in 1:num_steps
        local_error = local_error_bound(pf, terms, dt, norm)
        record!(acc, (; step, local_error))
    end
    return finalize!(acc)
end
