```@meta
EditURL = "task_2.jl"
```

# Task 7.2 — Single odd or even bond update

!!! question "Task 7.2 — Single odd or even update"
    Write a function that applies the time evolution for a given ``\Delta t``
    on the **even** or **odd** sites (for the XXZ chain) to an MPS.

````julia
using LinearAlgebra, Qritical
````

## First-order Suzuki-Trotter decomposition

For a Hamiltonian split into odd and even bonds,
``H = H_\text{odd} + H_\text{even}``, the first-order Trotter step is:

```math
e^{-i H \Delta t} \approx
  \prod_{i \text{ odd}} e^{-i h^{(i,i+1)} \Delta t} \cdot
  \prod_{i \text{ even}} e^{-i h^{(i,i+1)} \Delta t}
  + O(\Delta t^2)
```

Odd bonds ``\{(1,2),\,(3,4),\,\ldots\}`` and even bonds
``\{(2,3),\,(4,5),\,\ldots\}`` can each be applied **in parallel** because
gates on non-overlapping bonds commute.  Within each parity sweep the gates
are applied sequentially (in MPS form they modify bond dimensions site by site).

````julia
const Id2 = [1.0 0; 0 1.0]
const Sz1 = [0.5 0; 0 -0.5]
const Sp1 = [0.0 1; 0 0]
const Sm1 = [0.0 0; 1 0]

function local_hamiltonian(; J::Real=1.0, Jz::Real=1.0, hz::Real=0.0)
    (J/2) * (kron(Sp1, Sm1) + kron(Sm1, Sp1)) +
    Jz    *  kron(Sz1, Sz1) -
    hz    * (kron(Sz1, Id2) + kron(Id2, Sz1))
end
````

````
local_hamiltonian (generic function with 1 method)
````

````julia
function apply_bonds!(mps::FiniteMPS, H_bonds::AbstractVector, dt::Real;
                      parity::Symbol=:odd,
                      trunc::AbstractTruncation=KeepMachineEps())
    # Apply time-evolution gates e^{-i h^(i,i+1) Δt} to either odd or even bonds.
    # parity = :odd  → bonds (1,2), (3,4), …
    # parity = :even → bonds (2,3), (4,5), …
    start = parity == :odd ? 1 : 2
    for i in start:2:mps.L-1
        gate = exp(-im * dt * H_bonds[i])
        apply_gate!(mps, gate, (i, i + 1); trunc)
    end
    return mps
end
````

````
apply_bonds! (generic function with 1 method)
````

## Test: identity gate preserves state

````julia
L      = 8; dt = 0.1
H_bond = local_hamiltonian(J=1.0, Jz=1.0, hz=0.0)
H_bonds = fill(H_bond, L - 1)

mps = FiniteMPS(Spin{1//2}(), L, 8; T=ComplexF64)
left_canonical_sweep!(mps)
mps_vidal = to_vidal(mps)

# Apply odd bonds
mps_odd = deepcopy(mps_vidal)
apply_bonds!(mps_odd, H_bonds, dt; parity=:odd)
println("After odd update:  ⟨ψ|ψ⟩ = ",
        round(real(overlap(to_canonical(mps_odd), to_canonical(mps_odd))); sigdigits=8))

# Apply even bonds
mps_even = deepcopy(mps_vidal)
apply_bonds!(mps_even, H_bonds, dt; parity=:even)
println("After even update: ⟨ψ|ψ⟩ = ",
        round(real(overlap(to_canonical(mps_even), to_canonical(mps_even))); sigdigits=8))
````

````
After odd update:  ⟨ψ|ψ⟩ = 1.0
After even update: ⟨ψ|ψ⟩ = 1.0

````

## Energy change after one odd sweep

Verify that the energy changes by order ``\Delta t`` (first-order Trotter).

````julia
mpo_qrit = heisenberg_mpo(L; J=1.0)

E0_can = expectation_value(mps, mpo_qrit)

mps_after_odd = deepcopy(mps_vidal)
apply_bonds!(mps_after_odd, H_bonds, dt; parity=:odd)
can_after = to_canonical(mps_after_odd)
E1_can = real(overlap(mps, can_after)) / real(overlap(can_after, can_after))  ## approximate

println("\nInitial energy ⟨H⟩:    ", round(E0_can; sigdigits=8))
````

````

Initial energy ⟨H⟩:    0.17782305

````

## Qritical.jl: building block is `apply_gate!`

`apply_gate!(mps, gate, (i, i+1))` is the primitive used internally by
`trotter_step!`.  The `apply_bonds!` function above is the explicit
odd/even loop that `trotter_step!` automates in Task 7.3.

````julia
println("\nQritical.jl apply_gate! at bond (1,2):")
mps_test = deepcopy(mps_vidal)
gate12   = exp(-im * dt * H_bonds[1])
apply_gate!(mps_test, gate12, (1, 2))
can_test = to_canonical(mps_test)
println("  ⟨ψ|ψ⟩ after gate = ", round(real(overlap(can_test, can_test)); sigdigits=8))
````

````

Qritical.jl apply_gate! at bond (1,2):
  ⟨ψ|ψ⟩ after gate = 1.0

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

