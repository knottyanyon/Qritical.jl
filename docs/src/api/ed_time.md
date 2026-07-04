# ED Time Propagation

## Why exact time propagation?

Most time-evolution algorithms — TEBD, Runge-Kutta, Krylov methods — introduce some
form of approximation error. TEBD commits Trotter error: even with a tiny step
``\\Delta t``, the Suzuki-Trotter splitting ``e^{-i(H_A + H_B)\\Delta t} \\approx e^{-iH_A\\Delta t}\\,e^{-iH_B\\Delta t}``
is not exact, and that error accumulates over many steps. The exact-diagonalization
path does something completely different: it diagonalises ``H`` once and evaluates
``e^{-iHt}`` analytically. There is **no Trotter error** — only the floating-point
rounding of the diagonalisation itself, which is at machine-epsilon level.

The catch is cost. Diagonalising the full ``d^L \\times d^L`` matrix takes
``O(d^{3L})`` time and ``O(d^{2L})`` memory. For a spin-½ chain that is ``8^L``
and ``4^L``. This is feasible up to about ``L \\approx 12`` on a laptop, which is
exactly where Trotter-error cross-validation is most useful — small enough for ED
to be affordable, large enough to see the Trotter error scale properly.

## Real-time and imaginary-time propagation

**Real-time** propagation applies the unitary time-evolution operator:

```math
|\\psi(t)\\rangle = e^{-iHt}\\,|\\psi_0\\rangle.
```

Because ``H`` is Hermitian, ``e^{-iHt}`` is unitary, so the norm is exactly
preserved: ``\\|\\psi(t)\\| = \\|\\psi_0\\|`` for all ``t``. You can compute
time-dependent expectation values ``\\langle O(t) \\rangle = \\langle\\psi(t)|O|\\psi(t)\\rangle``
without any renormalisation.

**Imaginary-time** propagation replaces ``it`` with a real positive ``\\tau``:

```math
|\\psi(\\tau)\\rangle = e^{-H\\tau}\\,|\\psi_0\\rangle \\quad (\\text{unnormalised}).
```

Each eigencomponent ``|E_n\\rangle`` is suppressed by ``e^{-E_n\\tau}``. Because the
ground state has the smallest ``E_0``, it survives longest and the vector
``|\\psi(\\tau)\\rangle`` converges to the ground state as ``\\tau \\to \\infty``. The
rate of convergence is set by the **spectral gap** ``\\Delta = E_1 - E_0``: the
first excited state is suppressed relative to the ground state by a factor
``e^{-\\Delta\\tau}``. Larger gap means faster convergence.

Note that imaginary-time evolution is **non-unitary** — the norm decays.
`EDTimeResult.state` is returned unnormalised for imaginary time; call
`normalize(result.state)` to get the unit-norm ground-state approximation.

## The eigendecomposition trick

The computational trick that makes this efficient is to diagonalise ``H`` once:

```math
H = V\\,\\Lambda\\,V^\\dagger, \\qquad \\Lambda = \\mathrm{diag}(\\lambda_1, \\ldots, \\lambda_D).
```

Then the propagator at any time ``T`` is:

```math
e^{-iHT} = V\\,\\mathrm{diag}\\bigl(e^{-i\\lambda_1 T},\\,\\ldots,\\,e^{-i\\lambda_D T}\\bigr)\\,V^\\dagger.
```

Applying this to ``|\\psi_0\\rangle`` costs only ``O(d^{2L})``:

```math
|\\psi(T)\\rangle = V\\Bigl[\\mathrm{diag}\\bigl(e^{-i\\lambda_k T}\\bigr)\\cdot (V^\\dagger |\\psi_0\\rangle)\\Bigr].
```

The diagonalisation itself costs ``O(d^{3L})`` but you pay it once regardless
of how large ``T`` is or how many observables you compute afterwards.

## Cross-validation against TEBD

The most common use of ED time propagation is to check that your TEBD
implementation is correct and that the Trotter error scales as expected. The
workflow is:

1. Pick a small system, say ``L = 8`` spins.
2. Evolve with ED (no error) and with TEBD (Trotter error ``\\sim (\\Delta t)^p``).
3. Plot ``\\langle O(t) \\rangle_{\\text{ED}} - \\langle O(t) \\rangle_{\\text{TEBD}}``
   as a function of ``\\Delta t``. You should see it shrink as ``(\\Delta t)^p``.

If the error does not scale correctly, something is wrong with the Trotter
decomposition, not just the step size.

---

```@docs
StatevectorState
as_statevector
EDTimeResult
solve(::LatticeOperator, ::StatevectorState, ::ExactDiagonalization{:time}, ::ConstantProtocol)
```
