#ANCHOR - to review 1
# §8  TEBD evolution driver, Néel state, Tracker, and solve interface.
#
# `solve(H, Evolution(ψ₀), TEBD(...), protocol; tracker)` is the top-level entry
# point.  It hands off to the TEBD engine for each Trotter step and optionally
# records operator expectation values via the Tracker after each step.

# ----------------------------------------------------------------------------------------
# Néel state — alternating ↑↓ product state
# ----------------------------------------------------------------------------------------

"""
    neel_state(g::AbstractLayout; dof=SpinHalf()) -> FiniteMPS

Construct the Néel product state ``|\\uparrow\\downarrow\\uparrow\\downarrow\\cdots\\rangle``
on geometry `g`.

The state is a bond-dimension-1 MPS: site ``i`` carries ``|\\uparrow\\rangle`` for odd
``i`` and ``|\\downarrow\\rangle`` for even ``i``.  It has zero total ``S^z`` for even
``L``, zero entanglement entropy at every bond, and serves as the canonical initial
state for the XXZ quench problem (Ex 8).
"""
function neel_state(g::AbstractLayout; dof::AbstractDoF=SpinHalf())   # keyword arg with default `dof=SpinHalf()`; `AbstractLayout` and `AbstractDoF` are abstract supertypes that accept any concrete subtype
    L = length(sites(g))   # `sites(g)` returns a range 1:L; `length(...)` counts the sites; for Chain(4) this gives 4
    d = local_dim(dof)   # local Hilbert-space dimension: d=2 for SpinHalf
    # spin-½ basis: index 1 = |↑⟩, index 2 = |↓⟩
    tensors = Vector{QTensor}(undef, L)   # pre-allocate length-L vector of uninitialized QTensors    # L+1 bond spectra: one per boundary (L bonds → L+1 boundaries)
    bond_svs[1] = SingValSpectrum([1.0], 0.0, true)   # left boundary: trivial χ=1 spectrum; `[1.0]` = length-1 Float64 vector; `true` = normalised; SingValSpectrum stores singular values + truncation error + normalised flag
    bond_svs[L + 1] = SingValSpectrum([1.0], 0.0, true)   # right boundary: same trivial spectrum

    for i in 1:L   # build the rank-1 MPS tensor at each site
        data = zeros(ComplexF64, 1, d, 1)   # (χL=1, d, χR=1) tensor, all zeros; product state → bond dim 1 everywhere
        σ = isodd(i) ? 1 : 2    # 1 = ↑, 2 = ↓  # `isodd(i)` = true if i is odd. odd sites get spin-up (index 1), even sites get spin-down (index 2)
        data[1, σ, 1] = 1.0   # place a 1 in the σ-th physical slot; `data[1, σ, 1]` accesses element (1, σ, 1) — left bond index 1, physical index σ, right bond index 1 (1-indexed!)
        tensors[i] = QTensor(data, (upper(:vL, 1), upper(:σ, d), lower(:vR, 1)))   # wrap as QTensor with index variance tags: upper(contravariant) for left virtual and physical, lower(covariant) for right virtual
        bond_svs[i + 1] = SingValSpectrum([1.0], 0.0, true)   # bond i+1 has singular value = 1 (product state = rank-1 everywhere)
    end

    FiniteMPS(tensors, bond_svs, CanonicalForm(L + 1, 1), 0.0)   # construct the FiniteMPS: `CanonicalForm(L+1, 1)` marks it as right-canonical (ortho centre to the right of all sites, which for a product state is arbitrary); `0.0` = accumulated truncation error
end

# ----------------------------------------------------------------------------------------
# Evolution "study" type
# ----------------------------------------------------------------------------------------

"""
    Evolution(ψ₀)

A **study** in the [`DynamicsStudy`](@ref) regime: prepare the state `ψ₀`, propagate it,
observe.

`Evolution` is deliberately **schedule-agnostic**. It does not encode *how* the Hamiltonian
changes in time — a sudden switch, a slow ramp, or a periodic drive all use the same
`Evolution` study and differ only in the protocol handed to `solve`.

Used as the second argument to `solve`:

```julia
solve(H, Evolution(ψ₀), TEBD(formula, trunc), protocol)
```
"""
struct Evolution{Ψ} <: DynamicsStudy   # `struct T{Ψ} <: ParentType` — parametric struct; `Ψ` is the type parameter for the MPS state (e.g. `FiniteMPS`); `<: DynamicsStudy` makes this a subtype of DynamicsStudy — needed for dispatch
    ψ₀::Ψ   # the initial state; `::Ψ` annotates the field with the type parameter, so the struct is fully typed at compile time (zero-overhead field access)
end

# ----------------------------------------------------------------------------------------
# TEBD algorithm type
# ----------------------------------------------------------------------------------------

"""
    TEBD{F,T}

Algorithm type for time-evolving block decimation.

Fields:

  - `formula::F`  — the Trotter decomposition (e.g., `SuzukiTrotter(2)`).
  - `trunc::T`    — truncation strategy applied after each gate application.
"""
struct TEBD{F,T}   # parametric struct with two type parameters F (formula type) and T (truncation type); storing the type avoids boxing overhead — field accesses are direct memory reads
    formula::F   # Trotter decomposition formula (e.g. SuzukiTrotter{1} or SuzukiTrotter{2}); `::F` is a concrete type annotation for this field
    trunc::T   # truncation strategy (e.g. MaxBondDimTrunc(32) or NoTrunc()); `::T` makes `trunc` typed at compile time for zero-overhead dispatch
end

# ----------------------------------------------------------------------------------------
# Tracker + NoTracker
# ----------------------------------------------------------------------------------------

"""
    NoTracker

Null-object tracker that does not record any data.  Use when you only care about
the final state and not intermediate measurements.
"""
struct NoTracker end   # empty struct (zero-size type); represents "no tracking"; used as a null-object pattern. `struct T end` creates a singleton type

"""
    Tracker

Records operator expectation values during a time evolution.

# Constructor

```julia
Tracker(:name => operator, :name2 => operator2, …; every=1)
```

  - Keys are `Symbol` names; values are `LatticeOperator`s built via the standard constructors.
  - `every` (default 1): record every `every` steps.

Access recorded data via `result.measurements[:name]`.
"""
struct Tracker   # mutable state tracker for recording observables during evolution
    observables::Vector{Pair{Symbol,Any}}   # (name, LatticeOperator)  # `Vector{Pair{Symbol,Any}}` = vector of `:name => operator` pairs; `Pair{A,B}` = Python's `(a, b)` named pair; `Any` allows any LatticeOperator
    every::Int   # record every `every` steps (default 1 = every step)
end

Tracker(pairs::Pair{Symbol}...; every::Int=1) = Tracker(collect(pairs), every)   # `...` is Julia's splat (varargs; Python: `*args`); `Pair{Symbol}...` accepts any number of `:name => op` pairs; `collect(pairs)` converts the tuple to a Vector; `every::Int=1` is a keyword argument with default

# ----------------------------------------------------------------------------------------
# EvolutionResult
# ----------------------------------------------------------------------------------------

"""
    EvolutionResult

The return value of `solve(..., Evolution(...), TEBD(...), ...)`.

Fields:

  - `state::FiniteMPS`                             — the final MPS after all steps.
  - `steps::Int`                                   — the number of Trotter steps taken.
  - `measurements::Dict{Symbol, Vector{Float64}}`  — measurements recorded by the tracker.
"""
struct EvolutionResult   # result container for TEBD evolution; bundles state + metadata 
    state::FiniteMPS   # final MPS state after all nsteps Trotter steps
    steps::Int   # number of Trotter steps actually taken (= p.nsteps)
    measurements::Dict{Symbol,Vector{Float64}}   # `Dict{K,V}` = Python `dict`; Symbol keys = observable names; Vector{Float64} values = time series of measured expectation values
end

# ----------------------------------------------------------------------------------------
# solve dispatch: H + Evolution + TEBD + ConstantProtocol
# ----------------------------------------------------------------------------------------

"""
    solve(H, study::Evolution, algo::TEBD, p::ConstantProtocol;
          tracker=NoTracker(), log_every::Int=0) -> EvolutionResult

Evolve the initial state `study.ψ₀` under Hamiltonian `H` for `p.nsteps` Trotter steps
of size `p.dt` using the `algo.formula` Suzuki-Trotter decomposition.

At each step:

 1. Apply all single-bond gates from `trotter_steps(algo.formula, H, p.dt)`.
 2. For real-time evolution (`RealTime`), the norm is preserved automatically.
    For imaginary-time, the state is renormalized after the step.
 3. If the tracker fires at this step, record `⟨O⟩ / ⟨ψ|ψ⟩` for each observable.
 4. If `log_every > 0` and the step is a multiple of it, emit an `@info` line with
    the step count, maximum bond dimension, and largest per-bond truncation error
    so far — useful for watching long sweeps without a live dashboard.

Returns an `EvolutionResult` with the final state and all measurements.
"""
function solve(
    H::LatticeOperator,   # Hamiltonian: term list of on-site and bond interactions
    study::Evolution,   # study type: carries the initial state ψ₀
    algo::TEBD,   # algorithm type: carries the Trotter formula and truncation strategy
    p::ConstantProtocol;   # protocol: carries dt, nsteps, and the time axis (RealTime/ImaginaryTime); `;` separates positional from keyword arguments
    tracker=NoTracker(),   # optional tracker; default is NoTracker (do nothing); `=NoTracker()` creates a new NoTracker instance as the default
    log_every::Int=0,   # 0 = silent; positive = emit @info every `log_every` steps; `::Int` type annotation with default
)
    ψ = study.ψ₀   # extract initial state from the Evolution study; `study.ψ₀` field access
    mpo = Dict{Symbol,FiniteMPO}()   # `Dict{K,V}()` = empty typed dictionary ; will map observable name → MPO for the Tracker
    meas = Dict{Symbol,Vector{Float64}}()   # empty dict for time series; keys = observable names; values = Float64 arrays

    if tracker isa Tracker   # `isa` = isinstance 
        for (name, op) in tracker.observables   # destructure Pair: `name` = Symbol key, `op` = LatticeOperator value; iterates over all observables
            mpo[name] = MPO(op)   # build MPO for this observable once (outside the time loop for efficiency)
            meas[name] = Float64[]   # `Float64[]` = empty Float64 vector. will accumulate measurements over time steps
        end
    end

    for step in 1:p.nsteps   # main time-evolution loop; `1:p.nsteps` = range [1, 2, ..., nsteps]
        ψ = trotter_step(ψ, H, p.dt, algo.formula; trunc=algo.trunc, axis=p.axis)   # apply one Trotter step: implements e^{−i·dt·H} ≈ product of two-site gates; `axis=p.axis` tells trotter_step whether this is real- or imaginary-time

        if p.axis isa ImaginaryTime   # renormalize after imaginary-time step (imaginary-time evolution is not unitary → changes norm)
            ψ = canonicalize(ψ, LeftCanonical(algo.trunc))   # left-canonical sweep normalises and re-canonicalises the MPS; needed to prevent numerical instabilities from accumulating norm changes
        end

        if tracker isa Tracker && mod(step, tracker.every) == 0   # fire tracker every `every` steps; `mod(a, b)` = Python `a % b`; `&&` = AND (short-circuit)
            norm_sq = real(overlap(ψ, ψ))   # ‖ψ‖²; compute once per firing event and reuse for all observables
            for (name, op_mpo) in mpo   # iterate over all tracked observables
                push!(meas[name], real(expect(ψ, op_mpo)) / norm_sq)   # `push!(vec, elem)` appends ` discards imaginary rounding noise
            end
        end

        if log_every > 0 && mod(step, log_every) == 0   # optional logging: print progress every `log_every` steps
            χ_max = maximum(length(bs.values) for bs in ψ.bond_svs)   # maximum bond dimension currently in use; `maximum(gen)` = Python `max(gen)`; `length(bs.values)` = number of singular values = current bond dim at that bond
            ε_max = maximum(bs.ε for bs in ψ.bond_svs)   # maximum truncation error across all bonds (useful for diagnosing bond dim saturation)
            @info "TEBD step $step/$(p.nsteps)" χ_max ε_max   # `@info` emits a logging message at INFO level ; Julia's `Logging` module handles this; `$step` interpolates the variable; the extra args `χ_max ε_max` are appended as key=value pairs
        end
    end

    EvolutionResult(ψ, p.nsteps, meas)   # construct result struct: final MPS state, total steps taken, measurement dictionary
end
