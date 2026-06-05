```@meta
EditURL = "task_1.jl"
```

# Task 7.1 — Exponential of a local Hamiltonian

!!! question "Task 7.1 — Exponential of a local matrix"
    Write a function that receives the Hamiltonian parameters of an XXZ chain
    and returns the exponential ``e^{-i h^{(2)} \Delta t}`` of the local
    two-site Hamiltonian ``h^{(2)}``.

````julia
using LinearAlgebra, Qritical
````

## Two-site Hamiltonian

For a nearest-neighbour XXZ bond ``(i, i+1)``, the two-site Hamiltonian is

```math
h^{(2)} = \frac{J}{2}\left(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+\right)
         + J^z S_i^z S_{i+1}^z
         - h_z \left(S_i^z + S_{i+1}^z\right)
```

In the two-site basis ``\{|{\uparrow\uparrow}\rangle, |{\uparrow\downarrow}\rangle,
|{\downarrow\uparrow}\rangle, |{\downarrow\downarrow}\rangle\}``, this is a
``4 \times 4`` matrix built from Kronecker products of single-site spin operators.

````julia
# Spin-1/2 operators
const Id2 = [1.0 0; 0 1.0]
const Sz1 = [0.5 0; 0 -0.5]
const Sp1 = [0.0 1; 0 0]
const Sm1 = [0.0 0; 1 0]
````

````
2×2 Matrix{Float64}:
 0.0  0.0
 1.0  0.0
````

````julia
function local_hamiltonian(; J::Real=1.0, Jz::Real=1.0, hz::Real=0.0)
    # h^(2) for a single bond: 4×4 matrix in basis {↑↑, ↑↓, ↓↑, ↓↓}
    h = (J/2) * (kron(Sp1, Sm1) + kron(Sm1, Sp1)) +
        Jz    *  kron(Sz1, Sz1) -
        hz    * (kron(Sz1, Id2) + kron(Id2, Sz1))
    return h
end
````

````
local_hamiltonian (generic function with 1 method)
````

````julia
h2 = local_hamiltonian(J=1.0, Jz=1.0, hz=0.0)
println("Two-site Heisenberg h^(2):")
display(round.(h2; sigdigits=4))
println("\nEigenvalues: ", round.(sort(real(eigvals(h2))); sigdigits=5))
println("Expected:    [-0.75, 0.25, 0.25, 0.25]")
````

````
Two-site Heisenberg h^(2):

Eigenvalues: [-0.75, 0.25, 0.25, 0.25]
Expected:    [-0.75, 0.25, 0.25, 0.25]

````

## Gate: ``U = e^{-i h^{(2)} \Delta t}``

For real-time evolution, the two-site gate is the matrix exponential:

```math
U(\Delta t) = e^{-i h^{(2)} \Delta t}
```

Julia's `exp` works directly on matrices.

````julia
function make_gate(h::AbstractMatrix; dt::Real, imag_time::Bool=false)
    # Real time: U = exp(-i h Δt);  imaginary time: U = exp(-h τ)
    exponent = imag_time ? -dt : -im * dt
    return exp(exponent * h)
end
````

````
make_gate (generic function with 1 method)
````

````julia
dt = 0.1
U  = make_gate(h2; dt)

println("\nGate U = exp(-i h Δt) for Δt = $dt:")
println("  Unitarity check ‖U†U - I‖ = ", round(norm(U' * U - I(4)); sigdigits=4))
println("  det(U) = ", round(det(U); sigdigits=6), "  (should be ≈ 1)")
````

````

Gate U = exp(-i h Δt) for Δt = 0.1:
  Unitarity check ‖U†U - I‖ = 4.3e-16
  det(U) = 1.0 + 3.4694500000000004e-18im  (should be ≈ 1)

````

## Building bond gates for the full chain

For an L-site XXZ chain, each bond (i, i+1) can have different parameters.
We build a list of gates `H_bonds[i]` for bond ``i``.

````julia
function bond_hamiltonians(L::Int; J::Real=1.0, Jz::Real=1.0, hz::Real=0.0)
    h_bond = local_hamiltonian(; J, Jz, hz)
    return fill(h_bond, L - 1)   ## same h^(2) at every bond for uniform chain
end
````

````
bond_hamiltonians (generic function with 1 method)
````

````julia
L = 8
H_bonds = bond_hamiltonians(L; J=1.0, Jz=1.0, hz=0.0)
gates    = [make_gate(h; dt=0.05) for h in H_bonds]

println("\nChain of $L sites: $(length(gates)) two-site gates")
println("All gates unitary: ", all(norm(U' * U - I(4)) < 1e-12 for U in gates) ? "✓" : "✗")
````

````

Chain of 8 sites: 7 two-site gates
All gates unitary: ✓

````

## Qritical.jl note

Qritical.jl's `trotter_step!` and `time_evolve` accept `H_bonds` as a
`Vector` of two-site Hamiltonian matrices (not pre-exponentiated gates).
The exponential ``e^{-i h^{(2)} \Delta t}`` is computed internally by
`apply_gate!` each step.  You can verify this against `make_gate` above.

````julia
mps = FiniteMPS(Spin{1//2}(), L, 4; T=ComplexF64)
left_canonical_sweep!(mps)

mps_v = to_vidal(mps)

# Apply the gate at bond (1,2) manually and compare with apply_gate!
U12     = make_gate(H_bonds[1]; dt=0.05)
mps_v1  = deepcopy(mps_v)
apply_gate!(mps_v1, U12, (1, 2))
println("\napply_gate! on Vidal MPS: ⟨ψ|ψ⟩ after gate = ",
        round(real(overlap(to_canonical(mps_v1), to_canonical(mps_v1))); sigdigits=6))
````

````

apply_gate! on Vidal MPS: ⟨ψ|ψ⟩ after gate = 1.0

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

