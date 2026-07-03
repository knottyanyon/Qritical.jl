# §8  TEBD quench driver, Néel state, Tracker, and solve interface.
#
# `solve(H, Quench(ψ₀), TEBD(...), protocol; tracker)` is the top-level entry
# point.  It hands off to the TEBD engine for each Trotter step and optionally
# records operator expectation values via the Tracker after each step.

# ----------------------------------------------------------------------------------------
# Néel state — alternating ↑↓ product state
# ----------------------------------------------------------------------------------------

"""
    neel_state(g::AbstractGeometry; dof=SpinHalf()) -> FiniteMPS

Construct the Néel product state ``|\\uparrow\\downarrow\\uparrow\\downarrow\\cdots\\rangle``
on geometry `g`.

The state is a bond-dimension-1 MPS: site ``i`` carries ``|\\uparrow\\rangle`` for odd
``i`` and ``|\\downarrow\\rangle`` for even ``i``.  It has zero total ``S^z`` for even
``L``, zero entanglement entropy at every bond, and serves as the canonical initial
state for the XXZ quench problem (Ex 8).
"""
function neel_state(g::AbstractGeometry; dof::AbstractDoF=SpinHalf())
    L = length(sites(g))
    d = local_dim(dof)
    # spin-½ basis: index 1 = |↑⟩, index 2 = |↓⟩
    tensors = Vector{QTensor}(undef, L)
    bond_svs = Vector{SingValSpectrum}(undef, L + 1)
    bond_svs[1] = SingValSpectrum([1.0], 0.0, true)
    bond_svs[L + 1] = SingValSpectrum([1.0], 0.0, true)

    for i in 1:L
        data = zeros(ComplexF64, 1, d, 1)
        σ = isodd(i) ? 1 : 2    # 1 = ↑, 2 = ↓
        data[1, σ, 1] = 1.0
        tensors[i] = QTensor(data, (upper(:vL, 1), upper(:σ, d), lower(:vR, 1)))
        bond_svs[i + 1] = SingValSpectrum([1.0], 0.0, true)
    end

    FiniteMPS(tensors, bond_svs, CanonicalForm(L + 1, 1), 0.0)
end

# ----------------------------------------------------------------------------------------
# Quench "study" type
# ----------------------------------------------------------------------------------------

"""
    Quench{Ψ}

A study type that encodes a quantum quench: start from initial state `ψ₀` and
evolve under a Hamiltonian.

Used as the second argument to `solve`:

```julia
solve(H, Quench(ψ₀), TEBD(formula, trunc), protocol)
```
"""
struct Quench{Ψ}
    ψ₀::Ψ
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
struct TEBD{F,T}
    formula::F
    trunc::T
end

# ----------------------------------------------------------------------------------------
# Tracker + NoTracker
# ----------------------------------------------------------------------------------------

"""
    NoTracker

Null-object tracker that does not record any data.  Use when you only care about
the final state and not intermediate measurements.
"""
struct NoTracker end

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
struct Tracker
    observables::Vector{Pair{Symbol,Any}}   # (name, LatticeOperator)
    every::Int
end

Tracker(pairs::Pair{Symbol}...; every::Int=1) = Tracker(collect(pairs), every)

# ----------------------------------------------------------------------------------------
# QuenchResult
# ----------------------------------------------------------------------------------------

"""
    QuenchResult

The return value of `solve(..., Quench(...), TEBD(...), ...)`.

Fields:

  - `state::FiniteMPS`                             — the final MPS after all steps.
  - `steps::Int`                                   — the number of Trotter steps taken.
  - `measurements::Dict{Symbol, Vector{Float64}}`  — measurements recorded by the tracker.
"""
struct QuenchResult
    state::FiniteMPS
    steps::Int
    measurements::Dict{Symbol,Vector{Float64}}
end

# ----------------------------------------------------------------------------------------
# solve dispatch: H + Quench + TEBD + ConstantProtocol
# ----------------------------------------------------------------------------------------

"""
    solve(H, quench::Quench, algo::TEBD, p::ConstantProtocol; tracker=NoTracker()) -> QuenchResult

Evolve the initial state `quench.ψ₀` under Hamiltonian `H` for `p.nsteps` Trotter steps
of size `p.dt` using the `algo.formula` Suzuki-Trotter decomposition.

At each step:

 1. Apply all single-bond gates from `trotter_steps(algo.formula, H, p.dt)`.
 2. For real-time evolution (`RealTime`), the norm is preserved automatically.
    For imaginary-time, the state is renormalized after the step.
 3. If the tracker fires at this step, record `⟨O⟩ / ⟨ψ|ψ⟩` for each observable.

Returns a `QuenchResult` with the final state and all measurements.
"""
function solve(
    H::LatticeOperator, quench::Quench, algo::TEBD, p::ConstantProtocol; tracker=NoTracker()
)
    ψ = quench.ψ₀
    mpo = Dict{Symbol,FiniteMPO}()
    meas = Dict{Symbol,Vector{Float64}}()

    if tracker isa Tracker
        for (name, op) in tracker.observables
            mpo[name] = MPO(op)
            meas[name] = Float64[]
        end
    end

    for step in 1:p.nsteps
        ψ = trotter_step(ψ, H, p.dt, algo.formula; trunc=algo.trunc, axis=p.axis)

        if p.axis isa ImaginaryTime
            ψ = canonicalize(ψ, LeftCanonical(algo.trunc))
        end

        if tracker isa Tracker && mod(step, tracker.every) == 0
            norm_sq = real(overlap(ψ, ψ))
            for (name, op_mpo) in mpo
                push!(meas[name], real(expect(ψ, op_mpo)) / norm_sq)
            end
        end
    end

    QuenchResult(ψ, p.nsteps, meas)
end
