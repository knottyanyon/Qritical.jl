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
credits: N/A
=#

# SECTION -  AbstractErrorAccumulator: "how much do we trust it"
#
# Sibling to AbstractCollector, not a subtype - the same split collectors.jl draws between
# "what gets stored" and "how much do we trust it": neither interface depends on the other, so a
# caller can track truncation error alone with NoOpCollector() and pay nothing for spectrum
# storage, or vice versa.

"""
    AbstractErrorAccumulator

Root of the error-accumulator hierarchy: tracks a running numeric total (e.g. accumulated
truncation error) across an iterative routine, independently of whatever [`AbstractCollector`](@ref)
is or isn't doing. Every concrete subtype defaults to [`RecordingTrait`](@ref) [`Active`](@ref);
[`NoOpErrorAccumulator`](@ref) overrides this to [`Inactive`](@ref).
"""
abstract type AbstractErrorAccumulator end
RecordingTrait(::AbstractErrorAccumulator) = Active()

"""
    NoOpErrorAccumulator()

The universal opt-out for error accumulation - valid wherever any [`AbstractErrorAccumulator`](@ref)
is expected. [`record!`](@ref)/`finalize!` on a `NoOpErrorAccumulator` do nothing.
"""
struct NoOpErrorAccumulator <: AbstractErrorAccumulator end
RecordingTrait(::NoOpErrorAccumulator) = Inactive()

"""
    record!(acc::AbstractErrorAccumulator, ctx::NamedTuple)

Fold one step's contribution into the running error total.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:ctx]))
"""
function record!(acc::AbstractErrorAccumulator, ctx::NamedTuple)
    return record!(RecordingTrait(acc), acc, ctx)
end
record!(::Inactive, ::AbstractErrorAccumulator, ::NamedTuple) = nothing

"""
    finalize!(acc::AbstractErrorAccumulator; kwargs...)

Called once after the iterative routine completes; returns the accumulated total (in whatever
units the concrete accumulator tracks). A [`NoOpErrorAccumulator`](@ref) (or any [`Inactive`](@ref)
accumulator) returns `nothing`.
"""
function finalize!(acc::AbstractErrorAccumulator; kwargs...)
    return finalize!(RecordingTrait(acc), acc; kwargs...)
end
finalize!(::Inactive, ::AbstractErrorAccumulator; kwargs...) = nothing

# SECTION -  QuadratureTruncationErrorAccumulator: the concrete tracker Subroutines uses

"""
    QuadratureTruncationErrorAccumulator()

Tracks the running truncation error across a sweep via quadrature combination (`hypot`) of each
step's `ctx.ε` - the identity `ε² = Σᵢ εᵢ²` holds because `ε` is a 2-norm of discarded singular
values, so it's the *discarded weights* (the squares) that add across bonds, not the norms
themselves. Reads `ctx.ε` directly (never requires a `SingValSpectrum` to have been built), so
truncation-error tracking works even with the collector left as [`NoOpCollector`](@ref) - a
caller pays only for what the underlying factorization already returns for free.

`finalize!(acc; nrm=1.0)` returns the accumulated total as a global truncation error, optionally
normalized into a relative tolerance by the sweep's overall tracked norm `nrm`.

# Fields

  - `total   :: Float64`         - running quadrature-combined total
  - `history :: Vector{Float64}` - per-step raw `ε` values, in the order recorded
"""
mutable struct QuadratureTruncationErrorAccumulator <: AbstractErrorAccumulator
    total::Float64
    history::Vector{Float64}
end
function QuadratureTruncationErrorAccumulator()
    return QuadratureTruncationErrorAccumulator(0.0, Float64[])
end

function record!(::Active, acc::QuadratureTruncationErrorAccumulator, ctx::NamedTuple)
    acc.total = hypot(acc.total, ctx.ε)
    push!(acc.history, ctx.ε)
    return acc.total
end
function finalize!(::Active, acc::QuadratureTruncationErrorAccumulator; nrm::Real=1.0)
    return acc.total / nrm
end
