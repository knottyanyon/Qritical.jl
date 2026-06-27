# Disorder & Many-Body Localization

## Physics motivation

Most textbook spin chains are translationally invariant — every site looks the same.
Real materials are not.  Impurities, lattice defects, and nuclear spins all produce
**quenched disorder**: random on-site potentials ``h_i`` that are frozen in time.
Numerically we model this by replacing the uniform field ``h`` in the XXZ Hamiltonian
with a site-dependent random value drawn from a distribution, giving the **disordered
XXZ chain**:

```math
H = J \sum_{i} \mathbf{S}_i \cdot \mathbf{S}_{i+1}
  + J_z \sum_{i} S^z_i S^z_{i+1}
  - \sum_{i} h_i S^z_i
```

The fields ``h_i`` are drawn independently from a uniform distribution
``\mathcal{U}(-W, W)``, where ``W`` is the **disorder strength**.  Translational
symmetry is completely broken: no two sites are equivalent.

### The MBL transition

At weak disorder (small ``W``) the ground state looks much like the clean chain —
the spins are entangled across the whole system and the entanglement entropy scales
with the volume of the subsystem ("volume law").

Above a critical disorder strength ``W_c \approx 3.5`` (for the 1D XXZ chain at
``J = J_z = 1``), something remarkable happens: all eigenstates, not just the
ground state, become localised.  This is **many-body localization (MBL)**.  In the
MBL phase:

- Every eigenstate obeys an **area law** for entanglement — entropy saturates to a
  constant independent of system size.  This is unusual: for a generic excited
  state you would expect volume-law entanglement.
- Local observables retain memory of initial conditions forever; there is no
  thermalization.
- Because entanglement is low, MPS methods (and imaginary-time TEBD in particular)
  remain efficient even for excited states.

The transition at ``W_c`` separates the **ergodic/thermal phase** (small ``W``,
volume-law entanglement, thermalizing dynamics) from the **MBL phase** (large
``W``, area-law entanglement, localised dynamics).

### Why uniform disorder?

The uniform distribution ``\mathcal{U}(-W, W)`` is the standard choice in the
MBL literature because:

1. It has a single tunable parameter ``W`` that maps directly onto the disorder
   strength.
2. All values in ``[-W, W]`` are equally likely, so it places no prior bias on
   any particular field configuration.
3. Numerical results for ``W_c`` are well-established for this choice, making
   comparisons with the literature straightforward.

### Finding the ground state of a disordered model

The disordered Hamiltonian is still a nearest-neighbour model on a chain, so the
same `solve` interface works without any change.  For small systems (``L \lesssim 16``)
exact diagonalization gives the ground state exactly.  For larger systems,
imaginary-time TEBD converges especially quickly near the MBL transition because
the ground state has low entanglement.

A typical single-realization workflow:

```julia
using Random
L    = 8
W    = 3.5
rng  = MersenneTwister(42)
h    = disorder_realization(L, Uniform(-W, W), rng)
H    = XXZ(Chain(L); J=1.0, Jz=1.0, h=h)
res  = solve(H, GroundState(), ExactDiagonalization(:ground))
```

### Disorder averaging

A single realization is not physically meaningful on its own.  To extract
disorder-averaged quantities (e.g. the mean ground-state energy
``\langle E_0 \rangle`` or the average entanglement entropy) you repeat the
above loop over many seeds and average the observable at the end:

```julia
energies = [
    solve(
        XXZ(Chain(L); J=1.0, Jz=1.0,
            h=disorder_realization(L, Uniform(-W, W), MersenneTwister(s))),
        GroundState(), ExactDiagonalization(:ground)
    ).energy
    for s in 1:200
]
E_avg = sum(energies) / length(energies)
```

Using a different `MersenneTwister(s)` for each sample ensures the realizations
are statistically independent.  Using the same seed twice gives the same
realization — which is what you want when comparing two different algorithms on
the identical disorder landscape.

---

```@docs
Uniform
disorder_realization
```
