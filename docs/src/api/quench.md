# Quench & TEBD Solve Interface

## Physics motivation

A **quantum quench** is the simplest dynamical experiment you can run on a many-body
system: prepare the system in some initial state ``|\psi_0\rangle`` that is *not* an
eigenstate of the Hamiltonian ``H``, then let it evolve under ``H`` and watch what
happens. The energy is injected at time ``t = 0`` and the system explores its Hilbert space
through entanglement growth — this is what makes quenches such a useful diagnostic.

### The Néel state: a perfect starting point

The canonical initial state for the XXZ quench (Exercise 8) is the **Néel state**

```math
|\psi_0\rangle = |\uparrow\downarrow\uparrow\downarrow\cdots\rangle
```

Every site is in a definite ``S^z`` eigenstate, so the Néel state is a simple product
state — there is no entanglement between any pair of sites. As an MPS this means bond
dimension ``\chi = 1`` at every bond and zero entanglement entropy at every cut.
This is convenient not just computationally, but physically: it is as far from a ground
state as possible, and the subsequent entanglement growth is the phenomenon we want to
study.

The Néel state also has zero total ``S^z`` for an even chain length ``L``, which places
it in the zero-magnetisation sector where interesting dynamics happen.

### Connecting `solve` to TEBD: the typed interface

Qritical separates the *question* from the *method* through a typed interface:

```julia
solve(H, Quench(ψ₀), TEBD(formula, trunc), protocol; tracker = NoTracker())
```

- `H` is the Hamiltonian (an `LatticeOperator`).
- `Quench(ψ₀)` is the **study type** — it encodes "this is a quench starting from
  ``|\psi_0\rangle``". Having this as its own type means the same question can later be
  answered by different methods (TEBD, ED, TDVP) without changing the call site.
- `TEBD(formula, trunc)` is the **algorithm type** — it picks a Trotter decomposition
  and truncation strategy.
- `protocol` carries the time-step and number of steps; see the TEBD page for the full
  protocol machinery.

This separation is deliberate. It means `Quench` is meaningful to a physicist reading
the code ("we are doing a quench"), while `TEBD` is meaningful to a numerics person
("we are using TEBD to answer that question").

### Real time vs imaginary time

When the protocol uses `RealTime`, each Trotter gate is unitary:

```math
G = e^{-i\,\Delta t\,h_{ij}}
```

and the norm is exactly preserved — no renormalization needed.

When the protocol uses `ImaginaryTime` (``\Delta t \to -i\Delta\tau``), each gate is
instead a positive-semidefinite contraction:

```math
G = e^{-\Delta\tau\,h_{ij}}
```

This drives the state towards the lowest-energy component of ``|\psi_0\rangle``
(imaginary-time TEBD is a form of ground-state search), but at the cost of reducing the
norm ``\langle\psi|\psi\rangle`` at each step. The `solve` implementation re-canonicalizes
after every imaginary-time step to keep the MPS representation stable.

### Tracking observables

The `Tracker` records operator expectation values at each step without you needing to
write a loop. The measurement is the **Rayleigh quotient**

```math
\langle O \rangle = \frac{\langle\psi|O|\psi\rangle}{\langle\psi|\psi\rangle}
```

rather than a bare inner product, so the measurement is correct even when the norm has
drifted slightly during imaginary-time evolution. When you do not need any measurements,
pass `NoTracker()` (the default) and you pay no overhead.

---

## Types

### Initial states

```@docs
Quench
```

### Algorithms

```@docs
TEBD
```

### Tracking and results

```@docs
NoTracker
Tracker
QuenchResult
```

## Functions

### State preparation

```@docs
neel_state
```

### Time evolution

```@docs
solve(H::LatticeOperator, quench::Quench, algo::TEBD, p::ConstantProtocol)
```
