# TEBD & Suzuki-Trotter Time Evolution

## Physics motivation

**Time-Evolving Block Decimation (TEBD)** is the workhorse algorithm for time-evolving
an MPS under a nearest-neighbour Hamiltonian. The core idea is to approximate the
global time-evolution operator ``e^{-i\Delta t H}`` as a product of cheap two-site
gates — one gate per bond — using a **Suzuki-Trotter product formula**.

### Why the Trotter decomposition works

Suppose the Hamiltonian splits into even and odd bond groups:

```math
H = H_{\rm even} + H_{\rm odd}
```

Terms within the same group act on disjoint pairs of sites, so they commute
exactly: ``[H_{\rm even}, H_{\rm even}] = 0``.  This means the even-group exponential
factorises exactly into a product of independent two-site gates — no approximation
needed within a group.  The error comes only from the cross-group commutator
``[H_{\rm even}, H_{\rm odd}] \neq 0``.

The **first-order (Lie-Trotter)** formula uses this once:

```math
e^{-i\Delta t H} = e^{-i\Delta t H_{\rm even}}\,e^{-i\Delta t H_{\rm odd}} + O(\Delta t^2)
```

Error is ``O(\Delta t^2)`` per step, which means ``O(\Delta t)`` accumulated error over a
fixed total time ``T = N\Delta t``.

The **second-order (Strang / symmetric Trotter)** formula arranges the sub-steps as a
palindrome:

```math
e^{-i\Delta t H} =
  e^{-i\frac{\Delta t}{2} H_{\rm even}}\,
  e^{-i\Delta t H_{\rm odd}}\,
  e^{-i\frac{\Delta t}{2} H_{\rm even}}
  + O(\Delta t^3)
```

The leading commutator error cancels between the two half-steps by symmetry, leaving a
per-step error of ``O(\Delta t^3)`` — equivalent to ``O(\Delta t^2)`` accumulated error.
This is `SuzukiTrotter(2)`.

### Real time vs imaginary time

| Axis | Gate | Effect on norm |
|------|------|----------------|
| `RealTime` | ``G = e^{-i\Delta t\, h}`` | ``G^\dagger G = I`` — norm is exactly preserved |
| `ImaginaryTime` | ``G = e^{-\Delta\tau\, h}`` | ``G`` is Hermitian PSD, ``\|G\|\psi\rangle\| < \|\psi\|`` — must renormalize |

Imaginary-time TEBD converges to the ground state (the component that decays
slowest), analogously to the power method.  Renormalization is the caller's
responsibility after each call to `trotter_step`.

### Gate application: merge, contract, SVD split

Each two-site gate application follows four steps:

1. **Merge** the two adjacent MPS tensors into a single ``(\chi_L d) \times (d \chi_R)``
   matrix ``\Theta``.
2. **Contract** ``\Theta`` with the gate ``G[\sigma_1', \sigma_2', \sigma_1, \sigma_2]``,
   producing an updated ``\Theta'``.
3. **SVD** ``\Theta'`` and truncate singular values below threshold or beyond the target
   bond dimension.
4. **Store** the left-canonical ``A[\alpha, \sigma_1', r]`` tensor back at site ``i`` and
   absorb the singular-value diagonal into the right tensor ``B[r, \sigma_2', \beta]`` at
   site ``i+1``.

After the SVD the bond dimension at that bond is updated to ``r``.  Truncation error
accumulates across gates within a single Trotter step.

### On-site fields: the half-and-half split

A nearest-neighbour Hamiltonian often includes on-site terms like a magnetic field
``h_i S^z_i``.  Because every interior site sits at the boundary of two bonds, naively
assigning the on-site term to one bond would double-count it.  Instead,
`bond_hamiltonian` distributes each on-site field evenly across all bonds the site
participates in:

```math
h_{ij} = \sum_{(i,j)\text{ bond terms}} J_{ij}\,O_i \otimes O_j
        + \frac{1}{n_i}\,h_i\,O_i \otimes I_j
        + \frac{1}{n_j}\,h_j\,I_i \otimes O_j
```

where ``n_i`` is the number of bonds that site ``i`` belongs to (1 for boundary sites,
2 for interior sites on an open chain).  Boundary sites therefore receive their full
on-site contribution through the single bond they share; interior sites receive half
through each neighbouring bond.

---

```@docs
TimeAxis
RealTime
ImaginaryTime
Unitary
HermitianPSD
Propagator
opclass
gate
ConstantProtocol
total_time
bond_hamiltonian
apply_gate
TrotterSubstep
SuzukiTrotter
trotter_steps
trotter_step
```
