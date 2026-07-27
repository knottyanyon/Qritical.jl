# §6.3  Power method for ground-state search.
#
# Iterates  |ψₙ₊₁⟩ ∝ (λI − H)|ψₙ⟩  where λ is a positive shift large enough
# that the dominant eigenvalue of (λI − H) is λ − E₀ > 0.  After each
# application the state is normalised and recanonicalised by iterated SVD.
# Convergence is geometric in the ratio |ΔE / E_gap|.

"""
    PowerMethodResult

The return type of `power_method`. It bundles the converged (or best-so-far)
ground state together with the metadata you need to judge whether the result is
trustworthy.

## What the power method is doing

The power method is a discrete analogue of imaginary-time evolution. If you
propagate a state under ``e^{-\\tau H}`` for long enough, every excited component
decays faster than the ground-state component, and you are left with ``|\\psi_0\\rangle``.
The power method mimics this by repeatedly applying the operator

```math
\\hat{P} = \\lambda I - H
```

whose eigenvalues are ``\\lambda - E_k``. For ``\\hat{P}`` to be positive definite,
``\\lambda`` must exceed the **largest** eigenvalue ``E_{\\max}`` of ``H`` (not just the
ground-state energy ``E_0``). The dominant eigenvalue is then ``\\lambda - E_0`` because
``E_0 < E_k`` for all excited states, so ``\\lambda - E_0 > \\lambda - E_k``.
After ``n`` applications the overlap with any excited state ``|k\\rangle`` is suppressed by

```math
\\left(\\frac{\\lambda - E_k}{\\lambda - E_0}\\right)^n \\to 0
```

because the ratio in parentheses is strictly less than one. So iteration converges
geometrically in ``n``, with the convergence rate set by the spectral gap ``E_1 - E_0``.

## Fields

  - `state::FiniteMPS`: the final MPS, in mixed-canonical form centred at the middle bond.
    It is normalised to unit norm.
  - `energy::Float64`: the ground-state energy estimate, measured as the Rayleigh quotient
    ``\\langle\\psi|H|\\psi\\rangle / \\langle\\psi|\\psi\\rangle`` at the last iteration.
  - `converged::Bool`: `true` if ``|E_{n+1} - E_n| < \\text{tol}`` was satisfied before
    `maxiter` was reached. If `false`, treat the result with caution — the state may
    not have settled yet.
  - `iterations::Int`: the number of power iterations actually performed.
"""
struct PowerMethodResult   # immutable result container ; bundles the final MPS state and convergence metadata into one object
    state::FiniteMPS     # the converged (or best-so-far) MPS ground state; `::FiniteMPS` is the concrete type annotation (required for struct fields in Julia)
    energy::Float64      # Rayleigh-quotient estimate of E₀ = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩; stored as Float64 (64-bit float)
    converged::Bool      # `Bool` is Julia's boolean type (values: `true` and `false`, lowercase, not Python's `True`/`False`); true if energy change < tol before maxiter
    iterations::Int      # number of power iterations actually performed; useful for diagnosing slow convergence
end

"""
    power_method(H, ψ₀; shift, tol, maxiter, trunc) -> PowerMethodResult

Find the ground state of `H` by iterating ``|\\psi_{n+1}\\rangle \\propto (\\lambda I - H)|\\psi_n\\rangle``, where ``\\lambda`` is an energy shift large enough
to make ``\\lambda I - H`` positive definite. This is a discrete version of imaginary-time
evolution: each application amplifies the ground-state component relative to all excited
states, and after enough steps only the ground state survives.

## How `(λI − H)` is applied without a new MPO

Rather than building a separate MPO for ``\\lambda I - H`` at each iteration — which would
cost extra memory and compilation time — the implementation decomposes the action as

```math
(\\lambda I - H)|\\psi\\rangle = \\lambda|\\psi\\rangle - H|\\psi\\rangle
```

and evaluates the two terms separately:

 1. `Hψ = apply_mpo(mpo, ψ; trunc)` — zips the MPO ``H`` site by site over ``|\\psi\\rangle``,
    producing ``H|\\psi\\rangle`` as a new MPS whose bond dimension is at most
    ``\\chi_\\psi \\cdot \\chi_H`` before truncation.
 2. `ψ_new = add_mps(shift, ψ, -1.0, Hψ; trunc)` — forms the direct-sum MPS
    ``\\lambda|\\psi\\rangle + (-1) \\cdot H|\\psi\\rangle`` and immediately compresses it back
    to the target bond dimension via SVD.

## Energy measurement: the Rayleigh quotient

After each iteration the state is re-canonicalised by a `LeftCanonical` sweep, which
sweeps the norm weight rightward and absorbs it into the rightmost tensor. The resulting
MPS is *not* exactly unit-norm in the sense that `overlap(ψ, ψ) == 1` (the norm is
collected into one tensor rather than distributed). To get a reliable energy estimate
regardless of this, the code always measures the **Rayleigh quotient**:

```math
E = \\frac{\\langle\\psi|H|\\psi\\rangle}{\\langle\\psi|\\psi\\rangle}
```

This is why `expect` and `overlap` are called together rather than `expect` alone.

## Convergence

The iteration stops when the energy change between successive steps falls below `tol`:

```math
|E_{n+1} - E_n| < \\text{tol}
```

Convergence is geometric: each iteration multiplies the error by roughly
``(\\lambda - E_1)/(\\lambda - E_0)``, where ``E_1`` is the first excited-state energy.
A larger gap ``E_1 - E_0`` means faster convergence.

# Arguments

  - `H::LatticeOperator`: the Hamiltonian as a list of coupling terms.
  - `ψ₀::FiniteMPS`: the starting state. It need not be normalised; the first thing
    `power_method` does is centre-canonicalise it.
  - `shift::Real` (default: `4.0`): the energy shift ``\\lambda``. **This must exceed the
    largest eigenvalue ``E_{\\max}`` of ``H``**, not just ``E_0``. A safe upper bound is
    ``\\sum_i |\\text{coupling}_i|`` (sum of absolute coupling magnitudes); the default
    `4.0` is only adequate for chains with ``L \\lesssim 4``. If `shift` is too small,
    ``\\lambda I - H`` is not positive definite and the iteration will diverge.
  - `tol::Real` (default: `1e-8`): convergence threshold on the per-step energy change.
  - `maxiter::Int` (default: `200`): maximum number of power iterations before returning
    unconverged.
  - `trunc::AbstractTrunc` (default: `MaxBondDimTrunc(32)`): truncation scheme applied
    after `apply_mpo` and `add_mps` to keep bond dimensions bounded.

# Returns

  - `PowerMethodResult`: a struct with fields `state` (the final MPS), `energy`
    (the Rayleigh-quotient estimate of ``E_0``), `converged` (whether `tol` was met),
    and `iterations` (how many steps were taken).

# Extended help

**Choosing `shift`:** a common heuristic for the XXZ chain with coupling ``J`` is
``|\\lambda| = L \\cdot |J|``, where ``L`` is the chain length. For the Heisenberg
model at half-filling this is always sufficient. When in doubt, start with a larger
shift and check that the result does not depend on it.

**Bond dimension growth:** each power iteration roughly doubles the bond dimension of
the MPS before truncation (one factor from `apply_mpo`, one from `add_mps`). The
`trunc` argument cuts this back down after each step, so the wall-clock cost per
iteration scales as ``O(\\chi^3 d \\chi_{\\mathrm{mpo}})`` where ``\\chi`` is the target
bond dimension.

**When `converged == false`:** increase `maxiter`, tighten `trunc` (allow a larger bond
dimension), or check that `shift > |E_0|`.
"""
function power_method(
    H::LatticeOperator,
    ψ₀::FiniteMPS;   # `;` marks the end of positional arguments and start of keyword arguments; ψ₀ is a positional arg (subscript via Unicode: ψ₀ typed as \psi<Tab>\_0<Tab>)
    shift::Real=4.0,   # keyword arg with default: energy shift λ; `Real` accepts any real numeric type; default 4.0 is only safe for small chains
    tol::Real=1e-8,    # convergence tolerance on energy change; default 1e-8 is tight enough for most physics applications
    maxiter::Int=200,  # maximum iterations; `Int` = 64-bit integer on 64-bit systems
    trunc::AbstractTrunc=MaxBondDimTrunc(32),   # truncation strategy; `AbstractTrunc` is an abstract supertype; default caps bond dimension at 32
)
    mpo = MPO(H)   # build the MPO representation of H once; used in every iteration for H|ψ⟩; building it outside the loop avoids redundant work

    # Normalise initial state
    centre = div(1 + length(ψ₀.tensors), 2)   # compute the middle bond index; `div(a, b)` is integer division; `length(ψ₀.tensors)` = number of sites L; result is the bond nearest to the middle site
    ψ = canonicalize(ψ₀, BondCanonical(centre, trunc))   # put ψ₀ in bond-canonical form centred at the middle bond; this normalises the MPS and gives a well-defined starting state; `BondCanonical(centre, trunc)` is a struct encoding the target canonical form

    norm_sq(φ) = real(overlap(φ, φ))   # anonymous function : computes ‖φ‖² = ⟨φ|φ⟩; `real(...)` discards negligible imaginary part; used as denominator of Rayleigh quotient
    rayleigh(φ) = real(expect(φ, mpo)) / norm_sq(φ)   # another anonymous function: Rayleigh quotient E = ⟨φ|H|φ⟩/⟨φ|φ⟩; divides by norm_sq to handle unnormalised states safely; `expect(φ, mpo)` computes ⟨φ|H|φ⟩ via the MPO zipper sweep

    E_prev = rayleigh(ψ)   # initial energy estimate before any iterations; will be updated each step
    converged = false   # flag: starts false, set to true if tol criterion is met; `false` is Julia's boolean false
    iters = 0   # iteration counter; will track how many steps were actually performed

    for iter in 1:maxiter   # loop from 1 to maxiter inclusive; same semantics as Python's `for iter in range(1, maxiter+1)`
        iters = iter   # update the counter (so we have the right value after break)

        # Apply (λI - H): ψ_new = λψ - H|ψ⟩
        Hψ = apply_mpo(mpo, ψ; trunc=trunc)   # compute H|ψ⟩ as a new MPS; the bond dimension of Hψ is at most χ_mps × χ_mpo before truncation; `trunc=trunc` passes the keyword argument
        ψ_new = add_mps(shift, ψ, -1.0, Hψ; trunc=trunc)   # compute λ|ψ⟩ + (−1)·H|ψ⟩ = (λI−H)|ψ⟩; `add_mps(c1, ψ1, c2, ψ2)` forms the direct-sum MPS c1|ψ1⟩ + c2|ψ2⟩ and compresses; physics: (λI−H) amplifies the GS component if λ > E_max

        # Renormalise by full canonicalisation
        ψ = canonicalize(ψ_new, LeftCanonical(trunc))   # left-canonical sweep: sweeps norm weight from left to right, absorbing it into the rightmost tensor; this keeps the MPS numerically stable across iterations; `LeftCanonical(trunc)` encodes the target form

        # Measure energy as Rayleigh quotient (handles unnormalised ψ after sweep)
        E = rayleigh(ψ)   # compute energy estimate via Rayleigh quotient; the Rayleigh quotient E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ works even if ψ is not unit-normalised (which can happen after LeftCanonical)

        if abs(E - E_prev) < tol   # convergence check: stop if energy changed by less than tol; `abs(...)` = absolute value; physics: convergence means the power iteration has projected onto the GS
            converged = true   # mark as converged
            E_prev = E   # update for the final return value
            break   # exit the for loop immediately (same as Python `break`)
        end
        E_prev = E   # store current energy for comparison in the next iteration
    end

    # Return a properly normalized final state
    ψ_final = canonicalize(ψ, BondCanonical(div(1 + length(ψ.tensors), 2), trunc))   # put final state in bond-canonical form centred at the middle bond for clean normalisation; the middle-bond form is conventional for presenting the result
    PowerMethodResult(ψ_final, E_prev, converged, iters)   # construct result struct; Julia structs are constructed by calling the struct name like a function`)
end
