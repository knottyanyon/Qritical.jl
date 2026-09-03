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

Every concrete `(observable type, GaugeForm)` combination is its own method below.
`Hamiltonian`×{[`LeftCanonical`](@ref),[`RightCanonical`](@ref),[`MixedCanonical`](@ref)} are
implemented via [`_expectation_value_generic`](@ref); `Hamiltonian`×[`VidalGauge`](@ref) and every
`Correlator` combination remain stubs (see their own docstrings for why). A future single-site-
operator `Observable` subtype (e.g. `S_z` at one site) would add its own row to this matrix the
same way.
"""
function evaluate_expectation_value end

"""
    _expectation_value_generic(H::Hamiltonian, state::MPState) -> Scalar

`overlap(state, apply(to_mpo(H), state))` - `⟨state|H|state⟩` via the `apply`/`overlap` primitives
in `src/subroutines/contractions.jl`. Shared by the `LeftCanonical`/`RightCanonical`/
`MixedCanonical` methods below: nothing in `apply`/`overlap` reads `state`'s gauge tag, so all
three gauge forms genuinely share this one body (the "cheapest contraction direction per canonical
form" this file's own docstring anticipates is a real future optimization, not required for
correctness here - `apply` always touches every site regardless of gauge).
"""
function _expectation_value_generic(H::Hamiltonian, state::MPState)
    return overlap(state, apply(to_mpo(H), state))
end

function evaluate_expectation_value(
    H::Hamiltonian, state::MPState{LeftCanonical,S}
) where {S}
    return _expectation_value_generic(H, state)
end
function evaluate_expectation_value(
    H::Hamiltonian, state::MPState{RightCanonical,S}
) where {S}
    return _expectation_value_generic(H, state)
end
function evaluate_expectation_value(
    H::Hamiltonian, state::MPState{MixedCanonical,S}
) where {S}
    return _expectation_value_generic(H, state)
end
function evaluate_expectation_value(H::Hamiltonian, state::MPState{VidalGauge,S}) where {S}
    return _expectation_value_generic(H, _reconstruct_left_canonical(state))
end
function _expectation_value_generic(obs::LocalObservable, state::MPState)
    return local_expectation_value(state, obs.ops)
end

function evaluate_expectation_value(
    obs::LocalObservable, state::MPState{LeftCanonical,S}
) where {S}
    return _expectation_value_generic(obs, state)
end
function evaluate_expectation_value(
    obs::LocalObservable, state::MPState{RightCanonical,S}
) where {S}
    return _expectation_value_generic(obs, state)
end
function evaluate_expectation_value(
    obs::LocalObservable, state::MPState{MixedCanonical,S}
) where {S}
    return _expectation_value_generic(obs, state)
end
function evaluate_expectation_value(
    obs::LocalObservable, state::MPState{VidalGauge,S}
) where {S}
    return _expectation_value_generic(obs, _reconstruct_left_canonical(state))
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
