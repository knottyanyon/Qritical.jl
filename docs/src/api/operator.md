# Operators & Hamiltonians

A linear quantum-mechanical observable is a sum of weighted products of single-site
operators. The **[`Hamiltonian`](@ref)** is just one such observable — the operator that drives
dynamics. Magnetisation, density, and two-point correlators are others. Qritical builds
all of them the same way: as an `LatticeOperator` (a term list over a geometry and DoF), so the
same `expect(ψ, O)` routine measures energy, magnetisation, and ``\langle S^z_i S^z_j
\rangle`` without any special-casing.

The construction strategy is **lazy**: couplings and operator matrices are stored as a
list of `OneSiteTerm`s and `TwoSiteTerm`s; the dense matrix form (`matrix_repr`) and the [`MPO`](@ref Glossary#matrix-product-operator)
form (`MPO`) are computed on demand. This means you can build an `LatticeOperator`, inspect its
term list, modify couplings, and only pay for the [`MPO`](@ref Glossary#matrix-product-operator) contraction when you actually call
`expect`.

### Named constructors

The named constructors (`XXZ`, `Heisenberg`, `Ising`, `tV`, `kitaev_chain`) encapsulate
the physics sign conventions in one place. Every downstream computation — gates, MPO
build, Jordan–Wigner mapping — reads from the same term list, so a sign convention error
in the constructor propagates everywhere and is caught by the unit tests.

### Hamiltonian is an alias

`Hamiltonian = LatticeOperator`. The role (dynamics generator vs. observable) is determined by
how the instance is *used*, not by its type. `solve(H, GroundState(), DMRG(...))` uses
`H` as a generator; `expect(ψ, H)` uses the same object as an observable. A distinct
`Hamiltonian` type would add no information.

### Observable constructors

The observable constructors (`total_magnetization`, `staggered_magnetization`,
`op_at_site`, `two_site_op`) return ordinary `LatticeOperator`s. They plug directly into the
`Tracker` at each TEBD step via `expect(ψ, O)`, requiring no special measurement
infrastructure.

---

## Quick Reference

**Operator types:** [`OneSiteTerm`](@ref) · [`TwoSiteTerm`](@ref) · [`LatticeOperator`](@ref)

**Hamiltonian builders:** [`uniform_coupling`](@ref) · [`XXZ`](@ref) · [`Heisenberg`](@ref) · [`Ising`](@ref)

**Observables:** [`total_magnetization`](@ref) · [`staggered_magnetization`](@ref) · [`op_at_site`](@ref) · [`two_site_op`](@ref) · [`identity_operator`](@ref) · [`matrix_repr`](@ref)

---

## Types

### Terms and operators

```@docs
OneSiteTerm
TwoSiteTerm
LatticeOperator
```

## Functions

### Coupling patterns

```@docs
uniform_coupling
```

### Named Hamiltonians

```@docs
XXZ
Heisenberg
Ising
```

### Observable constructors

```@docs
total_magnetization
staggered_magnetization
op_at_site
two_site_op(g::AbstractGeometry, dof::AbstractDoF, opA::Symbol, iA::Int, opB::Symbol, iB::Int)
identity_operator
```

### Matrix representations

```@docs
matrix_repr(H::LatticeOperator)
```
