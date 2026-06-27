```@meta
EditURL = "task_3.jl"
```

# Task 7.3 — Time evolution by Trotter

!!! question "Task 7.3 — Time Evolution by Trotter"
    Write a function that applies a **single Trotter step** for a time
    evolution in the Suzuki-Trotter scheme to an MPS.

````julia
using LinearAlgebra, Qritical
````

## Full first-order Trotter step

A single Trotter step consists of one odd-bond sweep followed by one
even-bond sweep:

```math
\tilde{U}(\Delta t) =
  \underbrace{\prod_{i \text{ odd}}  e^{-i h^{(i,i+1)} \Delta t}}_{\text{odd}}
  \;\cdot\;
  \underbrace{\prod_{i \text{ even}} e^{-i h^{(i,i+1)} \Delta t}}_{\text{even}}
```

This approximates ``e^{-iH\Delta t}`` to first order in ``\Delta t``.
The Suzuki-Trotter error per step is ``O(\Delta t^2)``; over ``t/\Delta t``
steps the total error grows as ``O(\Delta t)``.

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

function apply_bonds!(mps, H_bonds, dt; parity=:odd, trunc=KeepMachineEps())
    start = parity == :odd ? 1 : 2
    for i in start:2:mps.L-1
        apply_gate!(mps, exp(-im * dt * H_bonds[i]), (i, i + 1); trunc)
    end
    return mps
end
````

````
apply_bonds! (generic function with 1 method)
````

````julia
function my_trotter_step!(mps::FiniteMPS, H_bonds::AbstractVector, dt::Real;
                          trunc::AbstractTrunc=KeepMachineEps())
    # First-order Trotter: odd bonds → even bonds
    # Caller is responsible for passing mps in VidalForm (e.g. via to_vidal).
    apply_bonds!(mps, H_bonds, dt; parity=:odd,  trunc)
    apply_bonds!(mps, H_bonds, dt; parity=:even, trunc)
    return mps
end
````

````
my_trotter_step! (generic function with 1 method)
````

## Verify: Néel state energy change matches first-order perturbation theory

For a Néel state ``|\psi_0\rangle = |{\uparrow\downarrow\uparrow\downarrow\cdots}\rangle``
and the Heisenberg chain, the energy per bond is
``\langle h^{(i,i+1)} \rangle = -J/4`` at each bond ``(i, i+1)``.
After one Trotter step ``\Delta t``:

```math
\langle H \rangle(t = \Delta t) - \langle H \rangle(0) \approx
  -i \Delta t \langle [H, H] \rangle + O(\Delta t^2) = O(\Delta t^2)
```

Since ``[H, H] = 0``, the energy change is zero to first order.
The actual change comes from the Trotter error ``O(\Delta t^2)``.

````julia
L    = 8; J = 1.0; dt = 0.02
H_bonds  = fill(local_hamiltonian(J=J, Jz=J), L - 1)
mpo_heis = heisenberg_mpo(L; J=J)

# Build Néel state
neel = FiniteMPS(Spin{1//2}(), L, 1; T=ComplexF64)
for i in 1:L
    e = isodd(i) ? ComplexF64[1, 0] : ComplexF64[0, 1]
    neel.tensors[i] = IndexedTensor(reshape(e, 1, 2, 1), neel.tensors[i].indices)
end
neel.form = ArbitraryForm()
left_canonical_sweep!(neel)

E0 = expectation_value(neel, mpo_heis)
println("Néel state energy ⟨H⟩₀ = ", round(E0; sigdigits=6))
````

````
Néel state energy ⟨H⟩₀ = -1.75

````

````julia
# Apply one Trotter step (student implementation)
neel_v = to_vidal(neel)
my_trotter_step!(neel_v, H_bonds, dt)
neel_after = to_canonical(neel_v)
left_canonical_sweep!(neel_after)

E1 = expectation_value(neel_after, mpo_heis)
println("After one step  ⟨H⟩₁ = ", round(E1; sigdigits=6))
println("ΔE = $(round(E1 - E0; sigdigits=4))  (should be O(Δt²) = O($(dt^2)))")
````

````
After one step  ⟨H⟩₁ = -1.75
ΔE = 1.1e-7  (should be O(Δt²) = O(0.0004))

````

## Entanglement grows under evolution

Starting from the product Néel state (entropy = 0 at every bond), the
Trotter evolution entangles the state — entropy should grow after each step.

````julia
S0 = entanglement_entropy(neel, L ÷ 2)
S1 = entanglement_entropy(neel_after, L ÷ 2)
println("\nEntropy at central bond:")
println("  S(t=0):  ", round(S0; sigdigits=4))
println("  S(t=Δt): ", round(S1; sigdigits=4), "  (should be > 0)")
````

````

Entropy at central bond:
  S(t=0):  -0.0
  S(t=Δt): 0.6931  (should be > 0)

````

## Qritical.jl equivalent: `trotter_step!`

Qritical.jl's `trotter_step!(mps, H_bonds, dt)` performs the same odd + even
sweep and is used as the building block in `time_evolve`.

````julia
neel_q = to_vidal(neel)
trotter_step!(neel_q, H_bonds, dt)   ## Qritical.jl version
E1_q   = expectation_value(to_canonical(neel_q), mpo_heis)

println("\nQritical.jl trotter_step! energy: ", round(E1_q; sigdigits=6))
println("Match with student: ", abs(E1 - E1_q) < 1e-10 ? "✓" : "✗ diff = $(abs(E1-E1_q))")
````

````

Qritical.jl trotter_step! energy: -1.75
Match with student: ✓

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

