# §6.3  Power method for ground-state search.
#
# Iterates  |ψₙ₊₁⟩ ∝ (λI − H)|ψₙ⟩  where λ is a positive shift large enough
# that the dominant eigenvalue of (λI − H) is λ − E₀ > 0.  After each
# application the state is normalised and recanonicalised by iterated SVD.
# Convergence is geometric in the ratio |ΔE / E_gap|.

struct PowerMethodResult
    state::FiniteMPS
    energy::Float64
    converged::Bool
    iterations::Int
end

"""
    power_method(H, ψ₀; shift, tol, maxiter, trunc) -> PowerMethodResult

Find the ground state of `H` by iterating ``|\\psi_{n+1}\\rangle \\propto
(\\lambda I - H)|\\psi_n\\rangle``.

# Arguments
- `H::Operator` — the Hamiltonian (term list).
- `ψ₀::FiniteMPS` — initial state (need not be normalised).
- `shift::Real` — the energy shift ``\\lambda``; must satisfy ``\\lambda > E_0``.
  A safe value is the sum of all absolute couplings in `H`.
- `tol::Real` — convergence threshold on the energy change per iteration.
- `maxiter::Int` — maximum number of iterations before giving up.
- `trunc::AbstractTrunc` — bond-dimension control applied after each `apply_mpo`.

# Returns
A `PowerMethodResult` with fields `state`, `energy`, `converged`, `iterations`.

# Notes
The method does **not** build a full MPO for ``(\\lambda I - H)`` at each step;
instead it applies ``\\lambda |\\psi\\rangle - H|\\psi\\rangle`` directly using
`apply_mpo` and `add_mps`, which keeps the bond dimension under control.

Convergence is measured as ``|E_{n+1} - E_n| < \\text{tol}``.
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
