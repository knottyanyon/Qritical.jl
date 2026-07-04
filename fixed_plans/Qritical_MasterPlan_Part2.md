# Qritical.jl — Master Plan, Part 2: Physical Systems & Symmetry

**Scope:** the physics-modelling half — geometry, degrees of freedom & basis changes, couplings, operators/Hamiltonian/observables, the named-model registry, and (because symmetries only have meaning once a concrete Hamiltonian exists) symmetry/TensorKit integration and anticommuting DoFs. The backend-agnostic tensor machinery (indices, `QTensor`, SVD, MPS, algorithms, `solve`) is **Part 1**.

> Split from the consolidated Master Plan v4 (2026-06-19) for context-window economy; content is verbatim, only reordered. Section numbers are retained from the master document, so cross-references like “§§6” / “§§12” / “§§23” point into Part 1.


**Project:** Qritical.jl — tensor-network computations in condensed-matter physics
**Status:** Single consolidated specification — Claude Code implementation guide and docstring/documentation source
**Revision:** v4 (master), 2026-06-19 — split into Part 1 (machinery) + Part 2 (physical systems)

> Supersedes Plan v2, the Fermions/Majoranas supplement, and v3. The degrees-of-freedom layer (§3) is
> rebuilt around the correct distinction: a **DoF is what the model defines** at each site (local Hilbert
> space + algebra + intrinsic statistics); a **basis change** is a separate, optional equivalence map
> between DoFs. This replaces the earlier "two-level system + representation" collapse, which could not
> express the Hubbard electron (4D) or spin-1 (3D). Every type/function carries a docstring-ready
> description. Code blocks are skeletons. TensorKit sector spellings are illustrative.

---


## 2. Layer 1 — Geometry

### `AbstractGeometry`
Abstract supertype for lattice geometries. A geometry answers two queries — its sites and its bonds —
which is all the Hamiltonian builder consumes.

### `Chain`
```julia
struct Chain <: AbstractGeometry
    L::Int            # number of sites, L ≥ 1
    periodic::Bool    # false = open boundary (default); true = periodic
end
Chain(L::Int) = Chain(L, false)
```
A 1D chain of `L` sites. Open boundaries by default: finite MPS/DMRG are formulated for open boundary
conditions; periodic boundaries inflate bond dimension and break canonical structure
(`MPS_review.pdf` §2.1), so `periodic = true` is supported but discouraged for the course.

### `sites`, `bonds`
```julia
sites(g::Chain) = 1:g.L
bonds(g::Chain) = g.periodic ? [(i, mod1(i+1, g.L)) for i in 1:g.L] :
                               [(i, i+1)            for i in 1:g.L-1]
```
`sites` returns site labels `1:L`. `bonds` returns nearest-neighbour `(i, j)` pairs (`L-1` consecutive
pairs for an open chain, plus the wrap bond if periodic). Longer-range connectivity is produced by
dedicated term generators (§5), not by `bonds`.

*Deferred:* `Square`, `Torus`, general `Lattice{V,E}`. `sites`/`bonds` is the seam they plug into.

---

## 3. Layer 2 — Degrees of freedom & basis changes

**The DoF is what the model places at each site:** its local Hilbert space, operator algebra, and
**intrinsic inter-site statistics** (commuting for spins and hard-core bosons; anticommuting for fermions
and Majoranas). Different models define genuinely different DoFs — a spin-½, a spin-1, a spinless fermion,
an electron (4D, Hubbard), a hard-core boson. A **basis change** is a *separate*, optional operation
(`basis_change`) that re-expresses a model in terms of a different DoF where an exact isomorphism exists
(Jordan–Wigner; `γ = c ± c†`). The statistics is intrinsic to the DoF and is what determines whether a
model needs a Jordan–Wigner string (dense) or fermionic grading (native, §13–14).

### `AbstractDoF` and concrete DoFs
```julia
abstract type AbstractDoF end

# --- Spins (commuting statistics) ---
struct Spin{S} <: AbstractDoF end             # S = 1//2, 1, 3//2, …; local_dim = 2S+1
const SpinHalf = Spin{1//2}
const SpinOne  = Spin{1}

# --- Fermions (anticommuting statistics) ---
struct SpinlessFermion <: AbstractDoF end     # 2D site {|0⟩,|1⟩};   c, c†, n
struct Electron        <: AbstractDoF end     # 4D site {|0⟩,|↑⟩,|↓⟩,|↑↓⟩} — the Hubbard DoF
struct Majorana        <: AbstractDoF end     # Majorana modes; realized on paired fermion sites

# --- Constrained bosons (commuting statistics) ---
struct HardCoreBoson   <: AbstractDoF end     # 2D site {|0⟩,|1⟩}; b, b†, n; no double occupancy
```
Each concrete DoF is the physical content a model defines per site. `Spin{S}` is parametrised by the spin
quantum number, giving local dimension `2S+1` (so the same type covers spin-½, spin-1, …). `Electron` is
the spinful fermion of the Hubbard model — a 4-dimensional site, **not** a two-level system. `Majorana` is
the from-the-start Majorana DoF; because a single Majorana has dimension √2, it is realized by pairing
modes onto complex-fermion sites (§M below). `HardCoreBoson` is a bosonic 2-level mode (no double
occupancy) — operators commute across sites, unlike the fermion.

### `local_dim`
```julia
local_dim(::Spin{S}) where {S} = Int(2S + 1)     # 2 for ½, 3 for 1, …
local_dim(::SpinlessFermion)   = 2
local_dim(::Electron)          = 4
local_dim(::HardCoreBoson)     = 2
local_dim(::Majorana)          = 2               # per paired (complex-fermion) site
```
Return the local Hilbert-space dimension of the DoF. Sizes dense/sparse representations and validates
tensor shapes.

### `statistics`
```julia
abstract type Statistics end
struct Commuting     <: Statistics end           # operators on different sites commute
struct Anticommuting <: Statistics end           # operators on different sites anticommute

statistics(::Spin)            = Commuting()
statistics(::HardCoreBoson)   = Commuting()
statistics(::SpinlessFermion) = Anticommuting()
statistics(::Electron)        = Anticommuting()
statistics(::Majorana)        = Anticommuting()
```
Return the DoF's **intrinsic** inter-site statistics. `Commuting` → ordinary tensor-product operators (no
string). `Anticommuting` → operators on different sites anticommute, realised either by Jordan–Wigner
(dense, via `basis_change` to a spin DoF) or by fermionic-graded spaces (native, §13). This single
function routes all sign handling — it is keyed on the DoF, not on a representation.

### `operators`
```julia
operators(::Spin{1//2}) = (; Sx, Sy, Sz, Sp, Sm, I)                    # 2×2
operators(::Spin{1})    = (; Sx, Sy, Sz, Sp, Sm, I)                    # 3×3 spin-1 matrices
operators(::SpinlessFermion) = (; c, cdag, n, I)                       # 2×2
operators(::Electron)        = (; cup, cdn, cupdag, cdndag,
                                  nup, ndn, n, Sz, Sp, Sm, I)          # 4×4 (intra-site ↑↓ JW order fixed)
operators(::HardCoreBoson)   = (; b, bdag, n, I)                       # 2×2, b² = 0
operators(::Majorana)        = (; γ1, γ2, I)                           # on the paired fermion site
```
Return the on-site operators for a DoF as a `NamedTuple`. Dense matrices in the natural basis; `TensorMap`s
over `S` in a graded backend (§6). All terms/gates/observables draw from this one source, so a convention
lives in one place. For an `Anticommuting` DoF these on-site matrices alone do not encode inter-site
anticommutation — that is supplied by Jordan–Wigner (dense) or sector braiding (native).

**Fixed `Electron` intra-site convention (standard).** The local basis is ordered `{|0⟩, |↑⟩, |↓⟩, |↑↓⟩}`
with the doubly-occupied state defined **spin-up-created-first**, `|↑↓⟩ ≡ c†↑ c†↓ |0⟩` — the convention used
by ITensor's `Electron` site, ALPS, and Essler et al., *The One-Dimensional Hubbard Model*. This single
choice fixes every intra-site sign; in particular `c†↓|↑⟩ = −|↑↓⟩`, equivalently `c↓|↑↓⟩ = −|↑⟩`, while
`c↑|↑↓⟩ = +|↓⟩`. In the ordered basis (columns/rows `1..4`):
```julia
# basis {|0⟩, |↑⟩, |↓⟩, |↑↓⟩}
cup = [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]      # c↑ : ⟨0|c↑|↑⟩=1, ⟨↓|c↑|↑↓⟩=+1
cdn = [0 0 1 0; 0 0 0 -1; 0 0 0 0; 0 0 0 0]     # c↓ : ⟨0|c↓|↓⟩=1, ⟨↑|c↓|↑↓⟩=-1   (the sign)
nup = cupdag*cup    # diag(0,1,0,1)
ndn = cdndag*cdn    # diag(0,0,1,1)
n   = nup + ndn     # diag(0,1,1,2)
F   = (-1)^n        # diag(1,-1,-1,1) — the fermion-parity (Jordan–Wigner string) operator
```
Inter-site anticommutation is carried by the parity operator `F`, inserted on odd operators when assembling
two-site terms/gates and the MPO (dense route), or supplied automatically by sector braiding (native route,
§13–14). Document this convention verbatim in the `Electron` docstring so the operator matrices and the
hopping terms cannot disagree on signs.

### `basis_change` — equivalence between DoFs
```julia
basis_change(H::Hamiltonian, target::AbstractDoF) -> Hamiltonian
```
Re-express a Hamiltonian over a *different* DoF, describing the identical operator. Defined only for DoF
pairs related by an exact isomorphism; errors otherwise.

| From ↔ To | Map | Inter-site string | Notes |
|---|---|---|---|
| `Spin{½}` ↔ `SpinlessFermion` | Jordan–Wigner | yes (statistics change) | 1D chain; NN terms local |
| `SpinlessFermion` ↔ `Majorana` | `γ = c ± c†` | n/a (same site algebra) | pairs per site |
| `Spin{½}` ↔ `HardCoreBoson` | direct (`b ↔ σ⁻`) | no (both commute) | string-free |
| `Electron` ↔ two `SpinlessFermion` species | per-spin Jordan–Wigner | yes | Hubbard → 2-flavour fermions/spins |
| `Spin{1}` ↔ fermions | no simple single-mode map | — | deferred (needs special constructions) |

This is what lets a model "defined with one DoF become an equivalent model through a basis change," and
`basis_change(H, Spin{1//2}())` is the **dense (Jordan–Wigner) route** for any fermionic model (§14).
On-site and NN anticommuting terms map to local terms; longer-range ones acquire an explicit `σᶻ`/string
(a multi-site term, wider MPO).

*Deferred:* the `Majorana` DoF's pairing-of-modes-into-sites detail (§M); `Electron`/Hubbard support;
`Spin{S>1//2}` operator tables beyond spin-1; `d`-level generalisation.

#### M. The `Majorana` DoF (note)
A single Majorana mode has dimension √2 — no tensor factor — so a `Majorana` model is realized by pairing
adjacent modes `γ₂ⱼ₋₁, γ₂ⱼ` onto a complex-fermion site (a 2D space). The `Majorana` DoF therefore lets a
model be *defined* in Majorana operators while the Hilbert space is built from paired fermion sites; the
pairing is the adjacent default (configurable later). `operators(Majorana())` returns `γ₁ = c+c†`,
`γ₂ = i(c†−c)` on that paired site (`γ₁=σˣ`, `γ₂=σʸ`, `{γₐ,γᵦ}=2δ`).

---

## 4. Layer 3 — Couplings

Couplings are **plain arrays** (Ex 6 takes `Jᵢ, Jᵢᶻ, hᵢ` as vectors). Uniform, site-dependent, and
disordered couplings are all "the array holds different values"; uniform is the constant-array case, so no
`"uniform"`/`"site-dependent"` flag is needed.

### `uniform`, `disorder_realization`
```julia
uniform(n::Int, x) = fill(x, n)
disorder_realization(n::Int, dist, rng = Random.default_rng()) = rand(rng, dist, n)
```
`uniform` returns a constant coupling array (constructors call it on scalar input). `disorder_realization`
draws one realization of `n` random couplings from `dist` (e.g. `Uniform(-W, W)`), taking an explicit
`rng` for reproducibility. Disorder is a sampling strategy for the same array slot; frustration is emergent
(connectivity + signs), not a flag.

---

## 5. Operators, the Hamiltonian, and observables

The central object is a **linear operator**: a sum of weighted products of single-site operators over a
geometry and DoF. The **Hamiltonian is just one instance** — the operator you evolve with — and
**observables** (magnetisation, density, two-point operators) are other instances, built exactly the same
way. The DoF type carries the statistics, so `MPO`/`physical_space`/`basis_change` read it directly.
`n`-body terms are anticipated; only 1- and 2-body are implemented for the course.

**The line that keeps this honest:** an `Operator` represents a *linear* observable, whose value is
`⟨ψ|O|ψ⟩`. Quantities that are *nonlinear* in the state — entanglement entropy `−Tr ρ log ρ`, the Schmidt
spectrum, bond dimension, truncation error — are **not** operator expectations and are computed by bespoke
functions of `ψ`, not via this abstraction (§11). Conflating the two would be a category error.

### `LocalTerm`, `BondTerm`
```julia
struct LocalTerm{O}
    site::Int
    op::O                 # single-site operator (matrix or TensorMap)
    coupling::Float64
end
struct BondTerm{O1,O2}
    i::Int; j::Int
    op_i::O1; op_j::O2     # need not be nearest-neighbour; the generic pair also covers pairing (cc, c†c†)
    coupling::Float64
end
```
A single-site contribution `coupling·op`, and a two-site contribution `coupling·(op_i ⊗ op_j)`.
Parametrised on operator types so stored operators match the active backend.

### `Operator` (and `Hamiltonian` as an instance)
```julia
struct Operator{D<:AbstractDoF, G<:AbstractGeometry, LT, BT}
    dof::D
    geom::G
    onsite::Vector{LT}    # LT a concrete LocalTerm{…}
    bond::Vector{BT}      # BT a concrete BondTerm{…}
    # plaquette::Vector{…}   # deferred: n-body terms
end
const Hamiltonian = Operator        # the Hamiltonian is the operator one evolves with — no separate type
```
A linear operator as a sum of stored terms over a DoF. The DoF type `D` records the algebra and (via
`statistics(D)`) the inter-site statistics — there is no separate "representation" field; the DoF *is* the
basis. Parametrised on concrete term-vector element types `LT, BT` for type stability (§19). Construction
is build-time, not hot-loop. `Hamiltonian` is an alias: the operator carries no notion of *role* — whether
it acts as the dynamics generator (passed to `solve`) or as an observable to measure (passed to a `Tracker`)
is decided by use, exactly as the operator carries no notion of temperature. `solve` may `@assert
ishermitian(H)` on the generator under a debug flag; a distinct Hermitian-enforcing wrapper is a later
option if needed, but the alias suffices.

### `expect`
```julia
expect(ψ, O::Operator) -> Number          # ⟨ψ|O|ψ⟩, the one contraction for any linear observable
```
Compute the expectation value of any operator in state `ψ` by contracting `⟨ψ| MPO(O) |ψ⟩` (reusing the §6
`MPO` converter; a direct contraction for a single-site `O` is a cheap special case). Energy is the
instance `expect(ψ, H)`; magnetisation, density, and two-point functions are others. This single routine is
what the Tracker's operator measurements call (§11).

---

## 7. Named constructors — the "registry"

Not a datastructure; exported constructors that fill term lists (cf. `LeftCanonical`/`RightCanonical`).
Each builds over its natural DoF; `basis_change` converts between DoFs. Scalars broadcast to constant
arrays; arrays pass through.

### Spin models (`Spin{½}`)
```julia
function XXZ(g::Chain; J = 1.0, Jz = 1.0, h = 0.0)
    nb = length(bonds(g)); L = g.L
    Jv, Jzv = (J isa Number ? uniform(nb,J) : J), (Jz isa Number ? uniform(nb,Jz) : Jz)
    hv  = h isa Number ? uniform(L,h) : h
    ops = operators(SpinHalf())
    onsite = [LocalTerm(i, ops.Sz, -hv[i]) for i in sites(g)]   # course convention: −Σ hᵢ Sᶻᵢ (Ex 3/6)
    bond = vcat(
        [BondTerm(i,j, ops.Sp, ops.Sm, 0.5Jv[b]) for (b,(i,j)) in enumerate(bonds(g))],
        [BondTerm(i,j, ops.Sm, ops.Sp, 0.5Jv[b]) for (b,(i,j)) in enumerate(bonds(g))],
        [BondTerm(i,j, ops.Sz, ops.Sz, Jzv[b])   for (b,(i,j)) in enumerate(bonds(g))],
    )
    Hamiltonian(SpinHalf(), g, onsite, bond)
end
Ising(g; J=1.0, h=0.0)      = …                          # transverse-field Ising (pin sign/field convention)
Heisenberg(g; J=1.0, h=0.0) = XXZ(g; J=J, Jz=J, h=h)
# heisenberg_spin1(g; J=1.0) = …  (Spin{1}, deferred)    # AKLT / Haldane-phase chain
```
`XXZ` builds the spin-½ XXZ chain (Ex 6 signature); couplings may be scalars or per-bond/site arrays.
`Heisenberg` is the isotropic case. A `Spin{1}` Heisenberg/AKLT constructor is a deferred extension.

### Fermionic models (`SpinlessFermion`)
```julia
function tV(g::Chain; t=1.0, V=0.0, μ=0.0)
    ops = operators(SpinlessFermion())                   # H = -t Σ(c†ᵢcⱼ+h.c.) + V Σ nᵢnⱼ - μ Σ nᵢ
    onsite = [LocalTerm(i, ops.n, -μ) for i in sites(g)]
    bond = vcat(
        [BondTerm(i,j, ops.cdag, ops.c, -t) for (i,j) in bonds(g)],
        [BondTerm(i,j, ops.c, ops.cdag, -t) for (i,j) in bonds(g)],
        [BondTerm(i,j, ops.n, ops.n,     V) for (i,j) in bonds(g)],
    )
    Hamiltonian(SpinlessFermion(), g, onsite, bond)
end
function kitaev_chain(g::Chain; t=1.0, Δ=1.0, μ=0.0)     # 1D p-wave topological superconductor
    ops = operators(SpinlessFermion())
    onsite = [LocalTerm(i, ops.n, -μ) for i in sites(g)]
    bond = vcat(
        [BondTerm(i,j, ops.cdag, ops.c,    -t) for (i,j) in bonds(g)],
        [BondTerm(i,j, ops.c,    ops.cdag, -t) for (i,j) in bonds(g)],
        [BondTerm(i,j, ops.c,    ops.c,    -Δ) for (i,j) in bonds(g)],   # pairing ΔN=-2
        [BondTerm(i,j, ops.cdag, ops.cdag, -Δ) for (i,j) in bonds(g)],   # pairing ΔN=+2
    )
    Hamiltonian(SpinlessFermion(), g, onsite, bond)
end
```
`tV` is the spinless t–V chain (conserves U(1)+parity); under `basis_change(_, Spin{1//2}())`
(Jordan–Wigner) it equals `XXZ` with **`J = 2t`, `Jz = V`, field from `−μ`** (plus a `V`-field shift and
constant). `kitaev_chain`'s pairing changes particle number by ±2, so it conserves **only parity ℤ₂** (use
`FermionParity()`, not `FermionNumber()`); at `t=Δ, μ=0` it hosts unpaired Majorana zero modes (a
ground-state doublet splitting `~e^{−L}`); under Jordan–Wigner it maps to the transverse-field Ising chain.

### Electron model (`Electron`) — Hubbard *(extension; the stated future target)*
```julia
function hubbard(g::Chain; t=1.0, U=0.0, μ=0.0)
    # H = -t Σ_{⟨ij⟩,σ}(c†_{iσ}c_{jσ}+h.c.) + U Σ_i n_{i↑}n_{i↓} - μ Σ_i n_i
    ops = operators(Electron())
    onsite = [LocalTerm(i, ops.nup * ops.ndn, U) for i in sites(g)] ∪
             [LocalTerm(i, ops.n, -μ)            for i in sites(g)]
    bond = vcat(
        [BondTerm(i,j, ops.cupdag, ops.cup, -t) for (i,j) in bonds(g)],
        [BondTerm(i,j, ops.cup, ops.cupdag, -t) for (i,j) in bonds(g)],
        [BondTerm(i,j, ops.cdndag, ops.cdn, -t) for (i,j) in bonds(g)],
        [BondTerm(i,j, ops.cdn, ops.cdndag, -t) for (i,j) in bonds(g)],
    )
    Hamiltonian(Electron(), g, onsite, bond)
end
```
The Hubbard model on a 4-dimensional electron site: hopping `t` per spin species, on-site repulsion `U`,
chemical potential `μ`. Conserves charge U(1), spin U(1) (`Sᶻ`), and parity, so the natural symmetric
backend is `U(1)_charge ⊠ U(1)_spin ⊠ parity`. With two species the Jordan–Wigner string bookkeeping is
unpleasant, so the **native fermionic-graded backend (§13–14) is strongly preferred** here — this is the
case that justifies Route B. *Deferred* (not needed for the course), but the `Electron` DoF and this
constructor are the seam for it.

### `majorana_operators`
```julia
majorana_operators(::SpinlessFermion) = let o = operators(SpinlessFermion())
    (; γ1 = o.c + o.cdag, γ2 = im*(o.cdag - o.c))
end
```
Return the Majorana pair on a complex-fermion site (`γ₁=σˣ`, `γ₂=σʸ` on one site). Equivalent to
`operators(Majorana())`. Site `j` carries Majoranas `2j−1, 2j`; `L` sites host `2L` Majoranas.

### Observable constructors (operators you measure)
The same machinery that builds Hamiltonians builds observables — they are all `Operator`s (§5), so they are
measured by `measure(O)` → `expect(ψ, O)` (§11). All return an `Operator`.
```julia
local_op(dof, sym::Symbol, site) =                     # a single-site observable, e.g. Sᶻ at `site`
    Operator(dof, geom_of(site), [LocalTerm(site, getproperty(operators(dof), sym), 1.0)], BondTerm[])

total_magnetization(g; dof = SpinHalf()) =             # M = Σ_i Sᶻ_i
    Operator(dof, g, [LocalTerm(i, operators(dof).Sz, 1.0) for i in sites(g)], BondTerm[])

staggered_magnetization(g; dof = SpinHalf()) =         # M_s = Σ_i (-1)^i Sᶻ_i  (Néel order parameter)
    Operator(dof, g, [LocalTerm(i, operators(dof).Sz, (-1.0)^i) for i in sites(g)], BondTerm[])

two_point(g, dof, opA::Symbol, iA, opB::Symbol, iB) =  # the operator Aᵢ Bⱼ (a single two-site term)
    Operator(dof, g, LocalTerm[],
             [BondTerm(iA, iB, getproperty(operators(dof),opA), getproperty(operators(dof),opB), 1.0)])

density(g) = total_magnetization(g; dof = SpinlessFermion())   # Σ_i nᵢ (reuse with the n operator)
```
- `local_op` / `total_magnetization` / `staggered_magnetization` — onsite-only observables; the staggered
  one is the Néel order parameter relevant to the Final's initial state.
- `two_point` — the *static* operator `AᵢBⱼ` whose expectation is an equal-time correlator
  `⟨AᵢBⱼ⟩`; the *time-dependent* correlator `⟨Aᵢ(t)Bⱼ⟩` is the `Correlator` **study** (§8), which evolves
  and uses these operators as its `A`, `B` fields.
- These compose freely and feed the Tracker each step: `Tracker(:mag => measure(total_magnetization(g)))`.

*Deferred:* registry datastructure (lookup tables / parameter-sensitivity metadata) atop the term→MPO
machinery.

---

## 13. Symmetry & TensorKit integration

Reconciles through the elementary space `S` (`physical_space(dof, sym)`) and each tensor's domain→codomain
partition. A `TensorMap` *is* a map from a domain to a codomain; giving tensors explicit per-leg spaces and
an explicit partition is the whole reconciliation. Native dense is the trivial-sector instance — so even the
ED backend (§9) is not a separate world but the one-block corner of the same framework.

### Concept map — objects
| Qritical concept | TensorKit concept |
|---|---|
| `Operator` / `Hamiltonian` (linear operator) | a `TensorMap : ℋ → ℋ` once formed |
| state / MPS | a `TensorMap : 𝟙 → ℋ` (factored into site tensors) |
| observable, gate | `TensorMap`s built from `operators(dof)` (gate = `exp` of a bond `TensorMap`) |
| DoF (`Spin{S}`, `SpinlessFermion`, `Electron`, …) | the local space + its sector *category* |
| `statistics(dof) == Anticommuting` | fermionic (super) grading → braiding supplies `−1` signs |
| `AbstractSymmetry` (`U1`,`SU2`,`FermionParity`,`FermionNumber`,…) | the sector *type* of the graded space |
| `physical_space(dof, sym)` | the elementary space `S` (objects of the sector category) |
| MPS site tensor | `TensorMap (left⊗phys) ← right`; partition = canonical-form **bipartition** (§state rule below) |
| MPO site tensor | `TensorMap (left⊗phys_in) → (phys_out⊗right)`; `phys_in`/`phys_out` split by **variance** |
| upper/lower index (`CovIndex`) | domain-vs-codomain placement + `V` vs `V'` duality (resolved below) |
| `LegPartition` | native realization of the `(codomain ← domain)` partition |
| backend dense vs symmetric/fermionic | trivial sector (≈ LinearAlgebra/`Matrix`,`sparse`) vs nontrivial (TensorKit) |

### Concept map — operations (the verbs)
| Qritical operation | TensorKit / Julia primitive |
|---|---|
| `expect`, overlap, MPO·MPS contractions | `@tensor`/`@planar` contraction; composition matches codomain↔domain |
| `do_svd` + `AbstractTrunc` (`ValCutoffTrunc`/`MaxBondDimTrunc`) | `tsvd` + `TruncationScheme`, **block-wise per sector** |
| canonical forms (`CanonicalizeSweep`) | `leftorth`/`rightorth` (QR / polar) per chosen partition |
| `gate(h, dt, axis)` | `exp` of a `TensorMap` (block-wise matrix exponential) |
| `ED(:ground)` / `DMRG` | `eigsolve` (KrylovKit Lanczos) on `sparse(H)` / effective `TensorMap` |
| `ED(:full)` | `eigen(Hermitian(Matrix(H)))` (trivial sector = one dense block) |
| ED time evolution | `exponentiate(sparse(H), …)` (KrylovKit) / dense `exp` |

Closure on the deferred items now that the index convention is fixed (§index-rule below): `CovIndex` is
exactly what a graded `TensorMap` already provides — an upper index is a codomain leg in `V`, a lower index a
domain leg in `V'`; there is no separate type to build, only the discipline of partitioning by variance
(operators) or bipartition (states). `LegPartition` is the native proto-`(codomain ← domain)`; aligning it
to TensorKit's partition removes the leg-grouping ambiguity behind the recurring SVD-destructuring
`BoundsError`. Both threads therefore *close into* the `TensorMap` partition rather than remaining parallel
machinery — which is the point of building on TensorKit at all.

These two threads are not abstract: they are **realized** by the index/leg layer of §23, which is already
implemented for the native backend (`TIx{Upper}`/`TIx{Lower}` = `CovIndex`; `MulTIx` + `Bipartition` +
`group_legs` = `LegPartition`). The only remaining step to *exploit* symmetry is to upgrade a leg's stored
space from a bare dimension (`ComplexSpace(dim)`) to a graded `ElementarySpace` (`Rep{U1}`, `Rep{SU2}`,
`Rep{FermionParity}`) — whose **sectors are the good quantum numbers**. Variance then selects irrep
(`Upper`/`V`) vs. dual irrep (`Lower`/`V'`), and an Einstein contraction (`Upper`↔`Lower`) pairs an irrep
with its dual, which *is* conservation of that quantum number, enforced by the block structure with no
explicit checking. So symmetry bookkeeping rides entirely on the leg's `(variance, space)` pair (§23).

### Index variance ↔ domain/codomain (von Delft covariant convention)
This package follows the covariant index notation of Altland & von Delft's *Mathematics for Physicists*
(chapters L5, L8, L10). The anchor is the (1,1)-tensor `A = eᵢ ⊗ eʲ Aⁱⱼ` (L10.3): the **upper** index `i`
is the *surviving output* index (the row of `Aⁱⱼ`, the component of the produced vector), and the **lower**
index `j` is the one *contracted with the input* `vʲ` (the column). Upper indices are contravariant and
live in the primal space `V` (`v = eⱼvʲ`, L256); lower indices are covariant and live in the dual `V'`
(`w = wᵢeⁱ`, L255). A `TensorMap`'s **codomain** is its output side — what the map *produces* — and its
**domain** is its input side — what it *consumes*. The correspondence is therefore:

| Index | Role | von Delft | TensorKit | Space |
|---|---|---|---|---|
| **upper** | ket-type — produced (output) | contravariant `vⁱ` | **codomain** (leg out of the map) | `V` (primal) |
| **lower** | bra-type — consumed (input) | covariant `wⱼ` | **domain** (leg into the map) | `V'` (dual) |

So an operator `O = Σ Oσ_{σ'} |σ⟩⟨σ'|` has upper `σ` = codomain (the ket it produces) and lower `σ'` =
domain (the ket it consumes); the MPO convention (§6) is written `phys_out` = upper = codomain,
`phys_in` = lower = domain to match. This mapping is read directly from index placement (upper vs. lower)
per von Delft's covariant rules — upper = output = codomain = primal `V`; lower = input = domain = dual `V'`.

**Crucial scope — this rule is for operators, not state coefficient tensors.** The variance rule above
adjudicates a leg's domain/codomain only when the leg has a *definite* bra/ket character — i.e. for
**operators** (MPO of `H`, observables, gates), which carry two physical legs of *opposite* character per
site (a bra/input leg, lower → domain; a ket/output leg, upper → codomain). A **state** coefficient tensor
`ψ^{σ₁…σ_L}` is different: every physical leg is the *same* kind of index (each `σᵢ` is a contravariant ket
component, all `Upper`), so there is no upper-vs-lower contrast among them to map. For a state, the
domain/codomain split at each reshape-and-SVD is simply the **Schmidt bipartition** — which legs are rows
vs. columns of the matricisation — a positional choice set by the canonicalisation sweep direction
(left-canonical groups the left block as codomain; right-canonical groups the right block). This is why the
MPS site-tensor entry above reads "*partition encodes canonical form*": it is the SVD cut, not the variance
rule. Conflating the two — applying the operator's upper→codomain rule to a state's uniform physical legs —
is a category error that would mis-wire every MPS tensor.

The discipline that matters is internal consistency: operator tensors assign physical legs by the
upper/lower variance rule; state tensors assign them by the SVD bipartition; both must agree at the
contraction seam (the state's ket leg, `Upper`/codomain, meets the operator's bra/input leg,
`Lower`/domain — an `Upper`↔`Lower` pairing), since a local inconsistency
(the same failure mode as the SVD-destructuring and fermion-sign traps) breaks contractions silently rather
than raising an error.

### What symmetry buys (precise)
Symmetry does **not** change the (at-worst linear) entanglement growth that causes the time wall. It makes
tensors block-diagonal in charge sectors; truncation keeps singular values per sector and never spends bond
budget on forbidden states — a prefactor gain that *pushes* the wall and compounds with the picture split.
XXZ conserves Sᶻ → U(1) (SU(2) isotropic). The **Hubbard model** carries the richest symmetry —
U(1)_charge ⊠ U(1)_spin ⊠ parity — which is exactly why its 4D sites benefit most from native grading.
(`MPS_review.pdf`; Kennes & Karrasch 2016.)

---

## 14. Anticommuting DoFs in detail (fermions & Majoranas)

This is the physics of the `Anticommuting` DoFs (`SpinlessFermion`, `Electron`, `Majorana`) and how signs
are realised. **The signs are intrinsic to the DoF**, not to a representation choice.

### The central problem: signs
On-site fermion/Majorana matrices do not encode inter-site anticommutation — naive use treats fermions as
hard-core bosons. The fix is one of two routes, by backend.

### Route A — Jordan–Wigner (dense): `basis_change(H, Spin{1//2}())`
`cⱼ = (∏_{k<j}(−σᶻ_k))σ⁻ⱼ`, etc. `nⱼ` is local; **NN hopping is local** (the XY exchange); longer-range
hopping keeps a `σᶻ` string (multi-site term). A fermionic Hamiltonian re-expressed over `Spin{½}` becomes
a spin model — exactly `tV → XXZ`, `kitaev_chain → Ising`. The course route (Ex 3/4), reusing spin
machinery. For `Electron`, two species require a combined per-spin string — workable but unpleasant.

### Route B — native fermionic grading (TensorKit)
A parity-graded space (`|0⟩` even, `|1⟩` odd; `c,c†` odd `TensorMap`s, `n,I` even). Category braiding supplies
every `−1` automatically — no string, no string-induced bond inflation, handles long-range, pairing, and
**multiple species** (Electron/Hubbard) uniformly. Correctness depends on a rigid
leg-order/`(codomain←domain)` convention (§13): **a wrong ordering gives a wrong braiding sign with no
error — only incorrect physics**. The Route-A↔Route-B agreement is the validation cross-check.

### Route comparison
| Aspect | A: Jordan–Wigner → spin (dense) | B: native graded (TensorKit) |
|---|---|---|
| Signs | manual `σᶻ` string | automatic via braiding |
| NN hopping/density | local; reuses `XXZ`/`Ising` | local |
| Long-range / pairing | `σᶻ` string → wider MPO | automatic |
| Multi-species (`Electron`/Hubbard) | combined per-spin string, awkward | automatic — **preferred** |
| Symmetry | spin U(1) | `FermionParity` (+U(1)_charge ⊠ U(1)_spin) |
| Fit | **course (Ex 3/4)** | thesis-scope (Hubbard, long-range, 2D) |

### Symmetry per model
| Model | DoF | Conserves | Backend symmetry |
|---|---|---|---|
| t–V | `SpinlessFermion` | U(1) + parity | `FermionNumber()` |
| Kitaev (pairing) | `SpinlessFermion` | parity ℤ₂ only | `FermionParity()` (U(1) broken) |
| Hubbard | `Electron` | charge + spin U(1) + parity | `U(1)_c ⊠ U(1)_s ⊠ parity` |
| any fermion via Route A | `Spin{½}` | spin U(1) | `U1()` after `basis_change` |

### Correctness notes
- *Parity superselection:* odd operators have zero expectation in parity eigenstates (`⟨c⟩=0` always);
  observables are parity-even (`⟨c†ⱼc_l⟩`, `⟨nⱼ⟩`). A vanishing `⟨c⟩` is not a bug.
- *Kitaev topological signature:* near-degenerate ground-state doublet splitting `~e^{−L}` at `t=Δ, μ=0`.
- *Electron intra-site order:* the convention is now fixed (§3) — `{|0⟩,|↑⟩,|↓⟩,|↑↓⟩}`, `|↑↓⟩=c†↑c†↓|0⟩`,
  giving `c↓|↑↓⟩=−|↑⟩`. Use those exact matrices and the parity operator `F`; cross-check #4 catches any drift.

### Validation cross-checks
```julia
# 1. Basis-change equivalence: t–V (fermion) ≡ XXZ (spin) under Jordan–Wigner.
g = Chain(10); Hf = tV(g; t=1.0, V=0.5); Hs = basis_change(Hf, SpinHalf())
@assert isapprox(energy(solve(Hf, GroundState(), DMRG(64,1e-10,20)).state),
                 energy(solve(Hs, GroundState(), DMRG(64,1e-10,20)).state); atol=1e-8)
# 2. Majorana algebra: γ₁=σˣ, γ₂=σʸ, {γₐ,γᵦ}=2δ.
m = majorana_operators(SpinlessFermion())
@assert m.γ1*m.γ2 + m.γ2*m.γ1 ≈ zeros(2,2) && m.γ1^2 ≈ I(2) && m.γ2^2 ≈ I(2)
# 3. Kitaev topological doublet.
@assert excited_gap(kitaev_chain(Chain(40); t=1.0, Δ=1.0, μ=0.0)) < 1e-6
# 4. Hubbard half-filling (extension): native-graded vs small-system ED agree.
H = hubbard(Chain(6); t=1.0, U=4.0); @assert isapprox(
    energy(solve(MPO(H; sym=:u1u1parity), GroundState(), DMRG(200,1e-10,30)).state),
    eigmin(Matrix(H)); atol=1e-8)
```

---
