# Power Method & Ground-State Search

The **power method** is one of the simplest algorithms for finding the ground state of a
quantum Hamiltonian. The idea comes from imaginary-time evolution: applying ``e^{-\tau H}``
to any state that has nonzero overlap with the ground state ``|\psi_0\rangle`` gives

```math
e^{-\tau H}|\phi\rangle = \sum_k c_k e^{-\tau E_k}|\psi_k\rangle
    \xrightarrow{\tau \to \infty} c_0 e^{-\tau E_0}|\psi_0\rangle
```

because the excited components ``e^{-\tau E_k}`` decay faster than the ground-state
component ``e^{-\tau E_0}`` whenever ``E_k > E_0``. The ground state dominates
exponentially fast.

For a converged ground state with first excited state above it, the rate of convergence is set by the [`spectral gap`](@ref Glossary#spectral-gap) ``\Delta = E_1 - E_0``.

For a small time step ``\tau``, the evolution operator expands as

```math
e^{-\tau H} \approx I - \tau H = \frac{1}{\lambda}\bigl(\lambda I - H\bigr)
    \quad \text{with } \lambda = \frac{1}{\tau}.
```

So applying ``(\lambda I - H)`` repeatedly is the discrete-time version of imaginary-time
evolution. After ``n`` steps the overlap with any excited state ``|k\rangle`` is suppressed
by the factor

```math
\left(\frac{\lambda - E_k}{\lambda - E_0}\right)^n
```

which is less than one whenever ``E_k > E_0`` and ``\lambda > E_0``. Convergence is
**geometric** in ``n``, with the rate set by the spectral gap ``E_1 - E_0``: a bigger gap
means faster convergence.

## Algorithm: MPS-native power iteration

At the MPS level, one power step does three things:

1. **Apply ``H``.**  `apply_mpo(H, ψ)` zips the MPO of ``H`` over ``|\psi\rangle``
   site by site, producing ``H|\psi\rangle`` as a new MPS. The bond dimension of the
   result grows as ``\chi_\psi \cdot \chi_H``.

2. **Form ``(\lambda I - H)|\psi\rangle``.**  Rather than building a separate MPO for
   ``\lambda I - H``, the code computes the direct sum

   ```math
   |\psi_{\rm new}\rangle = \lambda|\psi\rangle - H|\psi\rangle
   ```

   via `add_mps(λ, ψ, -1.0, Hψ)`. This direct-sum MPS has bond dimension at most
   ``2\chi_\psi + \chi_H`` before compression, and is immediately truncated by SVD back to
   the target ``\chi``.

3. **Re-canonicalise.** A `LeftCanonical` sweep normalises the state and stabilises
   subsequent iterations numerically.

The per-iteration cost is ``O(\chi^3 d \chi_{\rm mpo})`` where ``\chi`` is the target bond
dimension, ``d`` is the local Hilbert-space dimension, and ``\chi_{\rm mpo}`` is the MPO
bond dimension.

## The Rayleigh quotient for energy measurement

After the `LeftCanonical` sweep the MPS is in left-canonical form but the norm weight is
collected into the rightmost tensor rather than distributed uniformly. This means
`overlap(ψ, ψ)` is not exactly 1 from the perspective of all tensors simultaneously.
To get a reliable energy regardless, `power_method` always measures the **Rayleigh quotient**:

```math
E = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}
```

This is also correct at exact convergence, so using it throughout is safe and consistent.

## Convergence criterion

The iteration halts when the energy changes by less than `tol` between successive steps:

```math
|E_{n+1} - E_n| < \text{tol}
```

The `converged` field of `PowerMethodResult` tells you whether this was achieved within
`maxiter` steps.

---

## Quick Reference

**Types:** [`PowerMethodResult`](@ref)

**Functions:** [`power_method`](@ref)

---

## Types

### Results

```@docs
PowerMethodResult
```

## Functions

```@docs
power_method
```
