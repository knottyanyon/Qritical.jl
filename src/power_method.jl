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
struct PowerMethodResult
    state::FiniteMPS
    energy::Float64
    converged::Bool
    iterations::Int
end

"""
    power_method(H, ψ₀; shift, tol, maxiter, trunc) -> PowerMethodResult

Find the ground state of `H` by iterating ``|\\psi_{n+1}\\rangle \\propto
(\\lambda I - H)|\\psi_n\\rangle``, where ``\\lambda`` is an energy shift large enough
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
- `H::Operator`: the Hamiltonian as a list of coupling terms.
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
function power_method(H::Operator, ψ₀::FiniteMPS;
                      shift::Real    = 4.0,
                      tol::Real      = 1e-8,
                      maxiter::Int   = 200,
                      trunc::AbstractTrunc = MaxBondDimTrunc(32))

    mpo = MPO(H)

    # Normalise initial state
    centre = div(1 + length(ψ₀.tensors), 2)
    ψ = canonicalize(ψ₀, BondCanonical(centre, trunc))

    norm_sq(φ) = real(overlap(φ, φ))
    rayleigh(φ) = real(expect(φ, mpo)) / norm_sq(φ)

    E_prev = rayleigh(ψ)
    converged = false
    iters = 0

    for iter in 1:maxiter
        iters = iter

        # Apply (λI - H): ψ_new = λψ - H|ψ⟩
        Hψ    = apply_mpo(mpo, ψ; trunc=trunc)
        ψ_new = add_mps(shift, ψ, -1.0, Hψ; trunc=trunc)

        # Renormalise by full canonicalisation
        ψ = canonicalize(ψ_new, LeftCanonical(trunc))

        # Measure energy as Rayleigh quotient (handles unnormalised ψ after sweep)
        E = rayleigh(ψ)

        if abs(E - E_prev) < tol
            converged = true
            E_prev = E
            break
        end
        E_prev = E
    end

    # Return a properly normalized final state
    ψ_final = canonicalize(ψ, BondCanonical(div(1 + length(ψ.tensors), 2), trunc))
    PowerMethodResult(ψ_final, E_prev, converged, iters)
end
