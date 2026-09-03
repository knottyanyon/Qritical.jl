#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs: schollwoeck_2011
credits: N/A
=#

# The TEBD sweep loop: closes the last gap between `trotterize`'s gate sequence and an actual
# time-evolved `MPState` - walking the orthogonality center to each gate's site range, applying it
# via `apply_gate`, renormalizing, and reporting per-step diagnostics through one unified collector.

# SECTION -  TEBDAlgorithm - the run configuration

"""
    TEBDAlgorithm{PF<:ProductFormula}

Bundles everything a full TEBD run needs beyond the `Hamiltonian`/initial state themselves.

# Fields

  - `pf             :: PF`                - product formula (`LieTrotter`/`SuzukiTrotter`/`Suzuki4th`).
  - `bond_cutoff    :: Union{Int,Nothing}` - SVD truncation rank, forwarded to every `apply_gate` call.
  - `num_steps      :: Int`                - total number of outer Trotter steps.
  - `snapshot_every :: Int`                - cadence (in outer steps) at which `observables` get
    evaluated; `1` = every step, `0` = never. Independent of `TEBDStepSnapshot`'s cheap fields
    (`entanglement_entropy`/`truncation_error`), which are always computed.
  - `renormalize    :: Bool`               - whether to renormalize the state after each step
    (default `true` - standard TEBD convention, see [`evolve!`](@ref)).
"""
struct TEBDAlgorithm{PF<:ProductFormula}
    pf::PF
    bond_cutoff::Union{Int,Nothing}
    num_steps::Int
    snapshot_every::Int
    renormalize::Bool
end
function TEBDAlgorithm(
    pf::PF, bond_cutoff, num_steps::Int; snapshot_every::Int=1, renormalize::Bool=true
) where {PF<:ProductFormula}
    return TEBDAlgorithm{PF}(pf, bond_cutoff, num_steps, snapshot_every, renormalize)
end

# SECTION -  TEBDStepSnapshot - unified per-step collector payload

"""
    TEBDStepSnapshot{K,V}

One outer Trotter step's worth of tracked data, pushed to a `collector` via `step!` - the TEBD
analogue of `ExpectationValueSnapshot`/`SingValSpectrum`, bundling everything a step produces
rather than requiring a caller to wire up separate collectors/accumulators for each kind of data.

# Fields

  - `step                 :: Int`                      - outer step index (1-based).
  - `observables          :: Union{Dict{K,V},Nothing}` - from `evaluate_expectation_values`, or
    `nothing` if `observables` wasn't supplied to [`evolve!`](@ref) or this step isn't a
    `snapshot_every`-th step.
  - `entanglement_entropy  :: Float64`                  - at the post-step orthogonality center.
  - `truncation_error      :: Float64`                  - this step's local SVD truncation `ε`,
    summed across the step's `apply_gate` calls.
  - `trotter_error_bound   :: Union{Float64,Nothing}`   - this step's a priori Trotter bound, or
    `nothing` if no `trotter_norm` callback was supplied to [`evolve!`](@ref).
"""
struct TEBDStepSnapshot{K,V}
    step::Int
    observables::Union{Dict{K,V},Nothing}
    entanglement_entropy::Float64
    truncation_error::Float64
    trotter_error_bound::Union{Float64,Nothing}
end

# Julia's auto-generated inner constructor's `Union{Dict{K,V},Nothing}` field type can't infer
# K,V from a bare `nothing` argument - resolve explicitly rather than adding an ambiguous outer
# method.
_snapshot_kv(::Dict{K,V}) where {K,V} = (K, V)
_snapshot_kv(::Nothing) = (Any, Any)
function _make_snapshot(step::Int, observables, entropy, ε, bound)
    K, V = _snapshot_kv(observables)
    return TEBDStepSnapshot{K,V}(step, observables, entropy, ε, bound)
end

# SECTION -  internal helpers

# Entanglement entropy at state's own orthogonality center - an SVD isolating vL alone, then the
# same SingValSpectrum/entanglement_entropy primitives to_vidal's own bond-by-bond sweep uses.
function _center_entanglement_entropy(state::MPState)
    center = state.sites[state.orthogonality_center]
    t = tensor(center)
    iso = TensorKit.permute(t, ((1,), (2, 3)))   # vL ← (σ,vR)
    _, S, _, ε = factorize_tensor(iso, HasEntanglementSpectrum())
    svals = S.data
    normalized = isapprox(sum(abs2, svals), 1.0; atol=sqrt(eps(real(eltype(svals)))))
    return entanglement_entropy(SingValSpectrum(svals, ε, normalized))
end

# Divide the orthogonality-center site's tensor by a plain scalar - only the center site needs
# touching, mirroring `norm`'s own O(1) design.
function _rescale_center(state::MPState{G,S}, factor::Number) where {G,S}
    sites = copy(state.sites)
    c = state.orthogonality_center
    site = sites[c]
    sites[c] = QProcess(tensor(site) / factor, outputs(site), inputs(site))
    return TensorTrain{G,S,1}(sites, state.llim, state.rlim, c, state.ε)
end

# Walk the orthogonality center to (or adjacent to, for a 2-site gate) site_range, only
# re-canonicalizing when actually necessary.
function _walk_center(state::MPState, site_range::UnitRange{Int})
    target = first(site_range)
    already_there = if length(site_range) == 1
        state.orthogonality_center == target
    else
        state.orthogonality_center in (target, last(site_range))
    end
    return already_there ? state : canonicalize(state, MixedCanonicalize(target))
end

# SECTION -  evolve! - the TEBD sweep driver

"""
    evolve!(algorithm::TEBDAlgorithm, hamiltonian::Hamiltonian, state::MPState, dt::Float64;
            kind::Type{<:Time}=RealTime, observables::Union{Dict,Nothing}=nothing,
            trotter_norm::Union{Function,Nothing}=nothing,
            collector::AbstractCollector=NoOpCollector()) -> MPState

Evolve `state` under `hamiltonian` for `algorithm.num_steps` outer Trotter steps of size `dt`,
per `algorithm.pf`. Builds one [`TrotterStep`](@ref) up front (gates don't change step-to-step)
and, for each outer step, walks the orthogonality center to each gate's site range in order,
applies it via `apply_gate` (`bond_cutoff` forwarded), optionally renormalizes
(`algorithm.renormalize`, matching the standard TEBD convention that the discarded singular-value
weight is only a meaningful local error estimate when the state stays unit-norm between
truncations), and pushes one [`TEBDStepSnapshot`](@ref) per step into `collector` -
`entanglement_entropy`/`truncation_error` are always computed (cheap); `observables` are only
evaluated on `snapshot_every`-th steps (potentially expensive); `trotter_error_bound` is only
computed when `trotter_norm` (a `group -> Real` callback, per `trotter_error`'s own contract) is
supplied.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:tebd_hamiltonian, :tebd_state, :tebd_dt]))
"""
function evolve!(
    algorithm::TEBDAlgorithm{PF},
    hamiltonian::Hamiltonian,
    state::MPState,
    dt::Float64;
    kind::Type{<:Time}=RealTime,
    observables::Union{Dict,Nothing}=nothing,
    trotter_norm::Union{Function,Nothing}=nothing,
    collector::AbstractCollector=NoOpCollector(),
) where {PF}
    prop = propagator(hamiltonian, dt; kind=kind)
    step = trotterize(prop, algorithm.pf)

    state = canonicalize(state, MixedCanonicalize(1))

    for n in 1:algorithm.num_steps
        truncation_accumulator = QuadratureTruncationErrorAccumulator()

        for (site_range, gate) in step.block.gates
            state = _walk_center(state, site_range)
            state = apply_gate(
                state,
                gate,
                site_range;
                bond_cutoff=algorithm.bond_cutoff,
                accumulator=truncation_accumulator,
            )
        end

        if algorithm.renormalize
            state = _rescale_center(state, value(norm(state)))
        end

        step_ε = something(finalize!(truncation_accumulator), 0.0)
        state = MPState(
            state.sites,
            MixedCanonical(),
            state.llim,
            state.rlim,
            state.orthogonality_center,
            hypot(state.ε, step_ε),
        )

        entropy = _center_entanglement_entropy(state)
        trotter_bound = if trotter_norm === nothing
            nothing
        else
            trotter_error(step, hamiltonian, trotter_norm)
        end
        obs =
            if observables !== nothing &&
                algorithm.snapshot_every > 0 &&
                n % algorithm.snapshot_every == 0
                evaluate_expectation_values(observables, state)
            else
                nothing
            end

        step!(
            collector, (; snapshot=_make_snapshot(n, obs, entropy, step_ε, trotter_bound))
        )
    end

    return state
end

# SECTION -  evolve! - the Vidal-native (iTEBD-style) sweep driver, no center-walking needed

# Entanglement entropy directly off state.λs at `bond`, with no SVD - the whole point of Vidal
# gauge: every bond already carries its own Schmidt weights.
function _vidal_bond_entropy(state::MPState{VidalGauge,S}, bond::Int) where {S}
    state.λs === nothing && return 0.0
    svals = state.λs[bond].data
    normalized = isapprox(sum(abs2, svals), 1.0; atol=sqrt(eps(real(eltype(svals)))))
    return entanglement_entropy(SingValSpectrum(svals, 0.0, normalized))
end

"""
    evolve!(algorithm::TEBDAlgorithm, hamiltonian::Hamiltonian, state::MPState{VidalGauge,S}, dt::Float64;
            kind::Type{<:Time}=RealTime, observables::Union{Dict,Nothing}=nothing,
            trotter_norm::Union{Function,Nothing}=nothing,
            collector::AbstractCollector=NoOpCollector()) -> MPState{VidalGauge,S}

The classical Vidal (2003)/iTEBD sweep: same outer structure as the [`MixedCanonical`](@ref)
[`evolve!`](@ref) method, simplified since every bond already carries its own Schmidt weights -
gates apply directly in any order (no `MixedCanonicalize` center-walking), and there is no global
per-step renormalization (each 2-site [`apply_gate`](@ref) call already renormalizes its own
`λ_mid` locally, which is what keeps the whole Vidal-canonical chain normalized). `TEBDStepSnapshot`
reports `entanglement_entropy` at the bond the last 2-site gate in the step touched, read directly
off `state.λs` with no SVD at all.
"""
function evolve!(
    algorithm::TEBDAlgorithm{PF},
    hamiltonian::Hamiltonian,
    state::MPState{VidalGauge,S},
    dt::Float64;
    kind::Type{<:Time}=RealTime,
    observables::Union{Dict,Nothing}=nothing,
    trotter_norm::Union{Function,Nothing}=nothing,
    collector::AbstractCollector=NoOpCollector(),
) where {PF,S}
    prop = propagator(hamiltonian, dt; kind=kind)
    step = trotterize(prop, algorithm.pf)

    for n in 1:algorithm.num_steps
        truncation_accumulator = QuadratureTruncationErrorAccumulator()
        last_bond = 1

        for (site_range, gate) in step.block.gates
            state = apply_gate(
                state,
                gate,
                site_range;
                bond_cutoff=algorithm.bond_cutoff,
                accumulator=truncation_accumulator,
            )
            length(site_range) == 2 && (last_bond = first(site_range))
        end

        step_ε = something(finalize!(truncation_accumulator), 0.0)
        state = MPState{VidalGauge,S}(
            state.sites, 0, 0, nothing, hypot(state.ε, step_ε), state.λs
        )

        entropy = _vidal_bond_entropy(state, last_bond)
        trotter_bound = if trotter_norm === nothing
            nothing
        else
            trotter_error(step, hamiltonian, trotter_norm)
        end
        obs =
            if observables !== nothing &&
                algorithm.snapshot_every > 0 &&
                n % algorithm.snapshot_every == 0
                evaluate_expectation_values(observables, state)
            else
                nothing
            end

        step!(
            collector, (; snapshot=_make_snapshot(n, obs, entropy, step_ε, trotter_bound))
        )
    end

    return state
end
