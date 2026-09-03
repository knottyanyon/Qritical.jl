#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs:
credits: N/A
=#

# SECTION -  RecordingTrait: the shared opt-in/opt-out trait

"""
    RecordingTrait

Whether a recorder ([`AbstractCollector`](@ref) or `SimStudy.AbstractErrorAccumulator`) actually
does anything when called, decided at the type level (the same "Holy traits" style
`Qritical.Core`'s `SymmetryStructure`/`EntanglementStructure` already use) rather than a runtime
flag - this is what makes "skip building a spectrum nobody asked for" a zero-cost compile-time
branch rather than an `if` a caller pays for every bond regardless of the answer.

Concrete subtypes: [`Active`](@ref), [`Inactive`](@ref).
"""
abstract type RecordingTrait end

"""
The recorder actually runs.
"""
struct Active <: RecordingTrait end

"""
The recorder is a no-op - calling it does nothing and costs nothing beyond the trait dispatch itself.
"""
struct Inactive <: RecordingTrait end

# SECTION -  AbstractCollector: "what gets stored"

"""
    AbstractCollector

Root of the collector hierarchy: "what gets stored" during an iterative routine (a
canonicalization sweep, a time-evolution stepper, ...) that a caller may or may not want
recorded. Every concrete subtype gets a default [`RecordingTrait`](@ref) of [`Active`](@ref)
(a collector someone bothered to construct wants to run); [`NoOpCollector`](@ref) overrides this
to [`Inactive`](@ref) as the universal opt-out.

This is deliberately payload-agnostic: nothing here mentions singular values, `TIx`, or
`QProcess`. A caller in `Subroutines` builds whatever [`ctx`](@ref) `NamedTuple`(the recorded
context) it has on hand; a concrete collector only reads the fields its own [`step!`](@ref)
method looks at.
"""
abstract type AbstractCollector end
RecordingTrait(::AbstractCollector) = Active()

"""
    NoOpCollector()

The universal opt-out: valid wherever any [`AbstractCollector`](@ref) is expected, regardless of
what it would collect. [`step!`](@ref)/`finalize!` on a `NoOpCollector` do nothing.
"""
struct NoOpCollector <: AbstractCollector end
RecordingTrait(::NoOpCollector) = Inactive()

"""
    step!(collector::AbstractCollector, ctx::NamedTuple)

Entry point every call site uses to (maybe) record one step's worth of data.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:ctx]))
"""
step!(c::AbstractCollector, ctx::NamedTuple) = step!(RecordingTrait(c), c, ctx)
step!(::Inactive, ::AbstractCollector, ::NamedTuple) = nothing

"""
    finalize!(collector::AbstractCollector)

Called once after the iterative routine completes; returns whatever the collector considers its
result (e.g. an assembled table). A [`NoOpCollector`](@ref) (or any [`Inactive`](@ref) collector)
returns `nothing`.
"""
finalize!(c::AbstractCollector) = finalize!(RecordingTrait(c), c)
finalize!(::Inactive, ::AbstractCollector) = nothing
