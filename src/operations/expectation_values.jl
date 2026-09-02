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

# SECTION -  evaluate_expectation_value - double-dispatch skeleton

"""
    evaluate_expectation_value(observable::Observable, state::MPState) -> Scalar

`⟨state|observable|state⟩`, computed as a `QProcess`-formalism bra-observable-ket composition
(`dagger(state) ∘ observable_process ∘ state`, `src/processes/categorical.jl`) reducing to a
`Scalar`. Dispatches on both the observable's concrete type - does it need materializing to an MPO
first ([`Hamiltonian`](@ref)), or can it be contracted directly at the relevant legs
([`Correlator`](@ref))? - and `state`'s `GaugeForm` type parameter (`src/subroutines/gauge.jl`) -
which contraction direction is cheap for this canonical form (e.g. a zipper contraction sweeping
the direction a `LeftCanonical`/`RightCanonical` state is already isometric in)?

Every concrete `(observable type, GaugeForm)` combination is its own method below - **not yet
implemented** for any of them, pending real fields on [`Hamiltonian`](@ref)/[`Correlator`](@ref)
and the actual contraction logic. A future single-site-operator `Observable` subtype (e.g. `S_z`
at one site) would add its own row to this matrix the same way.
"""
function evaluate_expectation_value(::Hamiltonian, ::MPState{LeftCanonical,S}) where {S}
    return error(
        "evaluate_expectation_value(::Hamiltonian, ::MPState{LeftCanonical}) is not yet implemented",
    )
end
function evaluate_expectation_value(::Hamiltonian, ::MPState{RightCanonical,S}) where {S}
    return error(
        "evaluate_expectation_value(::Hamiltonian, ::MPState{RightCanonical}) is not yet implemented",
    )
end
function evaluate_expectation_value(::Hamiltonian, ::MPState{MixedCanonical,S}) where {S}
    return error(
        "evaluate_expectation_value(::Hamiltonian, ::MPState{MixedCanonical}) is not yet implemented",
    )
end
function evaluate_expectation_value(::Hamiltonian, ::MPState{VidalGauge,S}) where {S}
    return error(
        "evaluate_expectation_value(::Hamiltonian, ::MPState{VidalGauge}) is not yet implemented",
    )
end
function evaluate_expectation_value(::Correlator, ::MPState{LeftCanonical,S}) where {S}
    return error(
        "evaluate_expectation_value(::Correlator, ::MPState{LeftCanonical}) is not yet implemented",
    )
end
function evaluate_expectation_value(::Correlator, ::MPState{RightCanonical,S}) where {S}
    return error(
        "evaluate_expectation_value(::Correlator, ::MPState{RightCanonical}) is not yet implemented",
    )
end
function evaluate_expectation_value(::Correlator, ::MPState{MixedCanonical,S}) where {S}
    return error(
        "evaluate_expectation_value(::Correlator, ::MPState{MixedCanonical}) is not yet implemented",
    )
end
function evaluate_expectation_value(::Correlator, ::MPState{VidalGauge,S}) where {S}
    return error(
        "evaluate_expectation_value(::Correlator, ::MPState{VidalGauge}) is not yet implemented",
    )
end

# SECTION -  evaluate_expectation_values - batch evaluation with collector plug-in

"""
    ExpectationValueSnapshot{K,V}

One step's worth of computed expectation values, keyed identically to the `observables::Dict{K, <:Observable}` argument [`evaluate_expectation_values`](@ref) was called with. Plain payload
struct, playing the same role for a batch of expectation values that `SingValSpectrum`
(`src/subroutines/spectrum.jl`) plays for one bond's singular-value data during a canonicalization
sweep: built once per step, handed to an `AbstractCollector` via `step!`, and never itself decides
whether or how it's stored.

# Fields

  - `values :: Dict{K,V}` - observable key -> computed expectation value.
"""
struct ExpectationValueSnapshot{K,V}
    values::Dict{K,V}
end

"""
    evaluate_expectation_values(observables::Dict{K,<:Observable}, state::MPState;
                                 collector::AbstractCollector=NoOpCollector()) -> Dict{K}

Maps [`evaluate_expectation_value`](@ref) over every `(key, observable)` pair in `observables`,
collecting results into a `Dict` keyed identically to the input, and - when `collector` is
`Active` - pushes an [`ExpectationValueSnapshot`](@ref) built from that same `Dict` into
`collector` via `step!`. `collector` defaults to `NoOpCollector`, so a caller who only wants the
returned `Dict` (no tracking across steps) pays nothing extra, mirroring how `advance_bond!`
defaults its own `collector`/`accumulator` keywords. A future TEBD sweep can pass a concrete
collector here the same way it would for `SingValSpectrum`, to build up a
dataframe/dimensional-data-style table of tracked observables across time steps.
"""
function evaluate_expectation_values(
    observables::Dict{K,<:Observable}, state; collector::AbstractCollector=NoOpCollector()
) where {K}
    results = Dict{K,Any}(
        k => evaluate_expectation_value(obs, state) for (k, obs) in observables
    )
    if RecordingTrait(collector) isa Active
        step!(collector, (; snapshot=ExpectationValueSnapshot(results)))
    end
    return results
end
