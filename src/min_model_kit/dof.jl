# §3 / §6.1  Degrees-of-freedom layer.
#
# A DoF is the physical content placed at each lattice site. The behavior is determined by its local state space, operator algebra, and exchange statistics.  In particular, the exchange statistics is used to determine how signs of multi-site terms are handled (Eg: Jordan–Wigner string vs. native fermionic grading).

# ----------------------------------------------------------------------------------------
# Abstract DoF supertype
# ----------------------------------------------------------------------------------------

"""
    AbstractDoF

Abstract supertype to represent the physical degree of freedom place at a lattice site.

A degree of freedom type (DoF) captures the following crucial aspects of minimal model building:

 1. **Local Hilbert space** — the finite-dimensional space spanned by the on-site basis states (e.g. ``spin up \\{|{\\uparrow}\\rangle, spin down |{\\downarrow \\rangle\\}`` for a spin-½). Its dimension can be obtained with [`local_dim`](@ref) and is important in estimating bond dimension growth behavior.

 2. **Operator algebra** — the set of elementary matrices that act on that space, returned as a `NamedTuple` by [`algebra_generators`](@ref).

 3. **Exchange statistics** — whether operators on different sites
    commute ([`CCR`](@ref)) or anticommute ([`CAR`](@ref)),
    as returned by [`canonical_relation`](@ref).  This drives the choice between
    Jordan–Wigner string insertion and native fermionic grading.

Every concrete DoF in Qritical.jl is a subtype of `AbstractDoF`.

Defining each DoF as a **distinct Julia type** (rather than a symbol, integer tag, or
runtime enum) is a deliberate design choice that uses Julia's multiple-dispatch model.
When you call `algebra_generators(SpinHalf())`, Julia selects the correct method at
compile time based solely on the type — there is no runtime `if dof == :spin_half`
branch.  This means:

  - **Zero runtime overhead** — all dispatch is resolved by the compiler; the
    struct itself carries no data and occupies no memory.
  - **Extensibility** — adding a new DoF is just a new `struct` + new methods; no
    existing code needs touching.
  - **Type-safety** — passing the wrong DoF to a function is a compile-time error,
    not a silent wrong-answer bug.

Concrete subtypes: [`Spin`](@ref), [`SpinlessFermion`](@ref), [`Electron`](@ref),
[`MajoranaFermion`](@ref), [`HardCoreBoson`](@ref).

See also: [`local_dim`](@ref), [`canonical_relation`](@ref), [`algebra_generators`](@ref),
[`physical_space`](@ref)
"""
abstract type AbstractDoF end   # Julia abstract type: cannot be instantiated, but defines the "DoF" contract; all concrete DoF structs inherit from this; physics: each site carries a DoF that determines its local Hilbert space and operator algebra

# ----------------------------------------------------------------------------------------
# Concrete DoFs
# ----------------------------------------------------------------------------------------

"""
    Spin{S} <: AbstractDoF

A spin-``S`` degree of freedom site with local Hilbert-space dimension ``2S+1``,
where ``S`` is the **total spin quantum number** (spin magnitude): ``S = 0, \\tfrac{1}{2}, 1, \\tfrac{3}{2}, \\ldots``
— *not* the magnetic quantum number ``m_z``.

The type parameter `S` is a `Rational{Int}` (e.g. `1//2`, `1`, `3//2`).  Using a
rational rather than a `Symbol` or `Float64` serves two purposes: it allows Julia to
compute `local_dim = Int(2S + 1)` as a **compile-time constant** (no runtime cost),
and it expresses the half-integer nature of spin exactly, without floating-point
rounding.  `S` parameterises the *type* itself, so `Spin{1//2}` and `Spin{1}` are
two entirely distinct types that dispatch to different method implementations.

The local basis is ordered by **descending** magnetic quantum number:
``|m_z = S\\rangle, |m_z = S-1\\rangle, \\ldots, |m_z = -S\\rangle``.

Operators are built using the **Condon–Shortley phase convention** — the standard
choice (used in most quantum chemistry and condensed-matter textbooks) that fixes the
overall phase of each basis state so that all ladder-operator matrix elements are
**real and non-negative**.  Without this convention the matrices would only be
determined up to arbitrary state-dependent signs, making table comparisons
unreliable.  The resulting ladder operators satisfy

```math
S^{\\pm} |m\\rangle = \\sqrt{S(S+1) - m(m \\pm 1)} \\, |m \\pm 1\\rangle.
```

!!! tip "Convenience aliases"

    Prefer [`SpinHalf`](@ref) and [`SpinOne`](@ref) over writing `Spin{1//2}()` or
    `Spin{1}()` by hand — they are just `const` aliases, so there is no performance
    difference.

# Examples

```jldoctest
julia> local_dim(Spin{3//2}())
4

julia> local_dim(SpinHalf())
2
```
"""
struct Spin{S} <: AbstractDoF end       # S = 1//2, 1, 3//2, …; local_dim = 2S+1  # parametric struct `Spin{S}`: the spin quantum number S is baked into the TYPE (not a field); `Spin{1//2}` and `Spin{1}` are two completely different types; `1//2` is a Rational{Int} literal 

"""
    SpinHalf

Alias for `Spin{1//2}` — a two-level spin site with local Hilbert
space ``\\{|{\\uparrow}\\rangle, |{\\downarrow}\\rangle\\}`` and ``d = 2``.

# Examples

```jldoctest
julia> local_dim(SpinHalf())
2

julia> canonical_relation(SpinHalf()) isa CCR
true
```
"""
const SpinHalf = Spin{1//2}   # `const` makes this binding immutable. `= Spin{1//2}` creates a type alias — SpinHalf IS the type Spin{1//2}, not a value of that type

"""
    SpinOne

Alias for `Spin{1}` — a spin-1 site with three-dimensional local Hilbert space
``\\{|{+1}\\rangle, |{0}\\rangle, |{-1}\\rangle\\}`` and ``d = 3``.  Relevant for
Haldane-phase and AKLT physics.

# Examples

```jldoctest
julia> local_dim(SpinOne())
3
```
"""
const SpinOne = Spin{1}   # type alias for spin-1; used in Haldane chain models (AKLT state, etc.)

"""
    SpinlessFermion <: AbstractDoF

A spinless (single-orbital) fermionic site with local dimension ``d = 2``.

The only internal degree of freedom is **occupation** — whether the site holds a
particle (``|1\\rangle``) or is empty (``|0\\rangle``).  Spin is deliberately absent:
this DoF models situations where the spin degree of freedom is frozen out (e.g. a
fully spin-polarised band, or a model where only charge fluctuations matter).  It
is the minimal fermionic building block, and is directly useful for 1D tight-binding
or Kitaev-chain models.

To include spin, use [`Electron`](@ref) instead: that DoF tracks both spin-up and
spin-down occupancy, giving ``d = 4``, and naturally appears in Hubbard-type models.

The elementary operators are the annihilation operator ``c``
(``c|1\\rangle = |0\\rangle``) and the number operator ``n = c^\\dagger c``.

Because this is a fermionic site, [`canonical_relation`](@ref) returns [`CAR`](@ref),
meaning that operators on different sites anticommute:
``\\{c_i, c_j^\\dagger\\} = \\delta_{ij}``.
When working without native graded spaces, Jordan–Wigner string factors must be
inserted for non-adjacent operators.

See also: [`HardCoreBoson`](@ref) (same matrix structure, commuting statistics),
[`Electron`](@ref) (spin-½ version), [`algebra_generators`](@ref)
"""
struct SpinlessFermion <: AbstractDoF end   # 2D site {|0⟩,|1⟩}; c, c†, n  # empty struct (zero data, zero size); type identity alone determines dispatch; fermionic: canonical_relation returns CAR

"""
    Electron <: AbstractDoF

A spin-½ electron site (single orbital, Hubbard-model site) with local dimension
``d = 4``.

The local Hilbert space is spanned by four basis states in the
**"spin-up first"** (ITensor / Essler et al.) ordering:

```math
|0\\rangle, \\; |{\\uparrow}\\rangle, \\; |{\\downarrow}\\rangle, \\; |{\\uparrow\\downarrow}\\rangle
```

where ``|{\\uparrow\\downarrow}\\rangle \\equiv c^\\dagger_{\\uparrow} c^\\dagger_{\\downarrow} |0\\rangle``.
This ordering fixes the **intra-site sign convention**: because ``c_{\\downarrow}``
must anticommute past the already-present ``c_{\\uparrow}^\\dagger`` when acting
on the doubly-occupied state, one gets

```math
c_{\\downarrow} |{\\uparrow\\downarrow}\\rangle = -|{\\uparrow}\\rangle.
```

Inter-site statistics are [`CAR`](@ref) — both ``c_{\\uparrow}`` and
``c_{\\downarrow}`` are fermionic.

The full operator set (see [`algebra_generators(::Electron)`](@ref)) includes individual
spin-resolved annihilators `cup`/`cdn`, number operators `nup`/`ndn`, the total
number `n`, and spin operators `Sz`, `Sp`, `Sm`.

See also: [`SpinlessFermion`](@ref), [`algebra_generators`](@ref)
"""
struct Electron <: AbstractDoF end   # 4D site {|0⟩,|↑⟩,|↓⟩,|↑↓⟩}  # spin-½ electron (Hubbard site); d=4; fermionic (CAR)

"""
    MajoranaFermion <: AbstractDoF

A Majorana fermion site, realized on a paired (single complex-fermion) site with
local dimension ``d = 2``.

A Majorana operator ``\\gamma`` is its own Hermitian conjugate (``\\gamma^\\dagger = \\gamma``)
and satisfies the Clifford algebra ``\\{\\gamma_a, \\gamma_b\\} = 2\\delta_{ab}``.  In
Qritical.jl two Majorana modes are defined on each site via the decomposition of
a complex fermion ``c`` into real and imaginary parts:

```math
\\gamma_1 = c + c^\\dagger = \\sigma^x, \\qquad
\\gamma_2 = i(c^\\dagger - c) = \\sigma^y.
```

Both ``\\gamma_1`` and ``\\gamma_2`` are Hermitian, traceless, and square to the
identity on the local two-dimensional space.  The inter-site statistics are
[`CAR`](@ref), inherited from the underlying complex fermion.

See also: [`algebra_generators`](@ref)
"""
struct MajoranaFermion <: AbstractDoF end   # Majorana modes on a paired complex-fermion site  # two Majorana modes per paired site; d=2; fermionic (CAR)

"""
    HardCoreBoson <: AbstractDoF

A hard-core bosonic site with local dimension ``d = 2``.

Hard-core bosons obey the constraint ``b^2 = 0`` (at most one boson per site),
which gives a two-dimensional local Hilbert space ``\\{|0\\rangle, |1\\rangle\\}``.
Algebraically this looks identical to a spinless fermion site, but the crucial
difference is that bosonic operators on different sites **commute**:
``[b_i, b_j^\\dagger] = 0`` for ``i \\neq j``.

This means [`canonical_relation`](@ref) returns [`CCR`](@ref) and no Jordan–Wigner
strings are needed for non-adjacent operators.

See also: [`SpinlessFermion`](@ref) (anticommuting version), [`algebra_generators`](@ref)
"""
struct HardCoreBoson <: AbstractDoF end   # 2D site {|0⟩,|1⟩}; b, b†, n; b²=0  # hard-core bosons: same local matrices as SpinlessFermion but COMMUTING statistics (CCR, not CAR)

# ----------------------------------------------------------------------------------------
# CanonicalRelation — exchange statistics of the DoF
# ----------------------------------------------------------------------------------------

"""
    CanonicalRelation

Abstract supertype for the intrinsic inter-site statistics of a [`AbstractDoF`](@ref).

Statistics determines how operators on different sites combine:

  - [`CCR`](@ref): ``[O_i, O_j] = 0`` for ``i \\neq j``.  No sign factors needed.
  - [`CAR`](@ref): ``\\{O_i, O_j\\} = 0`` for ``i \\neq j``.  Signs must be
    tracked, either via Jordan–Wigner strings or via native fermionic grading.

**Why a separate type hierarchy?**
Statistics checks are on the hot path — every MPO bond term and every two-site gate
needs to know whether a Jordan–Wigner string is required.  Representing statistics as
a distinct `CanonicalRelation` type instead of a `Bool` field inside the DoF struct
means the check is resolved by **compile-time dispatch** rather than a runtime branch.
A function that accepts `::CAR` in its signature is only compiled for fermionic DoFs;
no dead code is generated for bosons.  It also keeps the statistics axis orthogonal to
the DoF axis: a function can dispatch on statistics alone (e.g. a Jordan–Wigner
string helper) without caring which specific DoF it was called with.

Use [`canonical_relation`](@ref) to query the statistics of any concrete `AbstractDoF`.
"""
abstract type CanonicalRelation end   # abstract type for exchange statistics; two concrete subtypes: CCR and CAR; used for compile-time dispatch 

"""
    CCR <: CanonicalRelation

Tag indicating that operators on different sites commute: ``[O_i, O_j] = 0``
for ``i \\neq j``.

Assigned to bosonic DoFs: [`Spin`](@ref) (all values of ``S``) and
[`HardCoreBoson`](@ref).  No Jordan–Wigner string is ever needed between
commuting sites.

See also: [`CAR`](@ref), [`canonical_relation`](@ref)
"""
struct CCR <: CanonicalRelation end   # operators on different sites commute  # Canonical Commutation Relations: [b_i, b_j†] = δ_ij (bosons); tag type with no fields — just identifies the statistics

"""
    CAR <: CanonicalRelation

Tag indicating that operators on different sites anticommute: ``\\{O_i, O_j\\} = 0``
for ``i \\neq j``.

Assigned to fermionic DoFs: [`SpinlessFermion`](@ref), [`Electron`](@ref), and
[`MajoranaFermion`](@ref).  When computing expectation values or building MPOs for
non-adjacent operators, a Jordan–Wigner string ``\\prod_k (-1)^{n_k}`` must be
inserted between sites ``i`` and ``j``.  In the future this will be
replaced by native graded-space arithmetic via TensorKit.

See also: [`CCR`](@ref), [`canonical_relation`](@ref)
"""
struct CAR <: CanonicalRelation end   # operators on different sites anticommute  # Canonical Anticommutation Relations: {c_i, c_j†} = δ_ij (fermions); tag type used to dispatch Jordan-Wigner string insertion

# ----------------------------------------------------------------------------------------
# local_dim — local Hilbert-space dimension
# ----------------------------------------------------------------------------------------

"""
    local_dim(dof::AbstractDoF) -> Int

Return the dimension of the local Hilbert space at a single site for degree of
freedom `dof`.

This number ``d`` sizes every dense array or sparse tensor that lives on a
physical leg: site tensors in an MPS have shape ``(\\chi_L, d, \\chi_R)``, and MPO
tensors have shape ``(\\chi_L, d, d, \\chi_R)``.  For spin-``S`` the formula is
``d = 2S + 1``, which is evaluated at compile time.

# Extended help / Dispatch notes

| DoF               | `local_dim` |
|:----------------- |:----------- |
| `Spin{S}`         | ``2S+1``    |
| `SpinlessFermion` | 2           |
| `HardCoreBoson`   | 2           |
| `MajoranaFermion` | 2           |
| `Electron`        | 4           |

# Examples

```jldoctest
julia> local_dim(SpinHalf())
2

julia> local_dim(SpinOne())
3

julia> local_dim(Spin{3//2}())
4

julia> local_dim(Electron())
4
```
"""
local_dim(::Spin{S}) where {S} = Int(2S + 1)   # computed at compile time  # `where {S}` captures the type parameter S; `Int(2S+1)` converts the rational 2S+1 to an integer at compile time (no runtime arithmetic); `::Spin{S}` is the argument pattern-matching on the type — no runtime dispatch overhead 
local_dim(::SpinlessFermion) = 2   # d=2: |0⟩, |1⟩
local_dim(::Electron) = 4   # d=4: |0⟩, |↑⟩, |↓⟩, |↑↓⟩
local_dim(::HardCoreBoson) = 2   # d=2: |0⟩, |1⟩
local_dim(::MajoranaFermion) = 2   # per paired (complex-fermion) site  # each paired site hosts two Majorana modes but has d=2 Hilbert space

# ----------------------------------------------------------------------------------------
# algebra_generators — on-site operator matrices as a NamedTuple
# ----------------------------------------------------------------------------------------

"""
    algebra_generators(dof::AbstractDoF) -> NamedTuple

Return a `NamedTuple` of ``d \\times d`` `ComplexF64` matrices for all elementary
on-site operators of `dof`.

These matrices are the building blocks of every Hamiltonian and observable in
Qritical.jl: [`XXZ`](@ref), [`Heisenberg`](@ref), [`Ising`](@ref), and the
observable constructors all retrieve entries from this tuple by name.

# Extended help / Dispatch notes

**`Spin{1//2}` (`SpinHalf`)** — ``d = 2``, basis ``\\{|{\\uparrow}\\rangle, |{\\downarrow}\\rangle\\}``
(``m_z = +\\tfrac{1}{2}`` first):

  - `Sz` — ``S^z = \\tfrac{1}{2}\\,\\mathrm{diag}(+1,-1)``
  - `Sp` — ``S^+ = \\begin{pmatrix}0&1\\\\0&0\\end{pmatrix}``, raises ``m_z``
  - `Sm` — ``S^- = (S^+)^\\dagger``
  - `Sx` — ``S^x = (S^+ + S^-)/2``
  - `Sy` — ``S^y = (S^+ - S^-)/(2i)``
  - `I`  — ``2\\times 2`` identity

**`Spin{1}` (`SpinOne`)** — ``d = 3``, basis ``\\{|{+1}\\rangle, |{0}\\rangle, |{-1}\\rangle\\}``
(Condon–Shortley convention, same fields as above but ``3\\times 3``).

**`SpinlessFermion`** — ``d = 2``, basis ``\\{|0\\rangle, |1\\rangle\\}`` (vacuum first):

  - `c`    — annihilator: ``c|1\\rangle = |0\\rangle``
  - `cdag` — ``c^\\dagger = c^T``
  - `n`    — number operator: ``n = c^\\dagger c = \\mathrm{diag}(0,1)``
  - `I`    — identity

**`HardCoreBoson`** — ``d = 2``, same matrix structure as `SpinlessFermion` but
field names `b`, `bdag`, `n`, `I`.  Statistics are [`CCR`](@ref).

**`Electron`** — ``d = 4``, basis ``\\{|0\\rangle, |{\\uparrow}\\rangle, |{\\downarrow}\\rangle, |{\\uparrow\\downarrow}\\rangle\\}``
("spin-up first" convention):

  - `cup`, `cupdag` — spin-up annihilator/creator
  - `cdn`, `cdndag` — spin-down annihilator/creator; note ``c_{\\downarrow}|{\\uparrow\\downarrow}\\rangle = -|{\\uparrow}\\rangle``
  - `nup`, `ndn`, `n` — partial and total number operators
  - `Sz`, `Sp`, `Sm` — spin operators built from ``c^\\dagger_\\uparrow c_\\downarrow`` etc.
  - `I`  — ``4\\times 4`` identity

**`MajoranaFermion`** — ``d = 2``, two Hermitian MajoranaFermion operators:

  - `γ1 = c + c†` (equals ``\\sigma^x`` on the Fock site)
  - `γ2 = i(c† - c)` (equals ``\\sigma^y``)
  - `I`  — identity

# Examples

```jldoctest
julia> ops = algebra_generators(SpinHalf());

julia> ops.Sz
2×2 Matrix{ComplexF64}:
 0.5+0.0im   0.0+0.0im
 0.0+0.0im  -0.5+0.0im

julia> ops.Sp
2×2 Matrix{ComplexF64}:
 0.0+0.0im  1.0+0.0im
 0.0+0.0im  0.0+0.0im
```
"""
function algebra_generators(::Spin{1//2})   # method dispatches on the singleton type Spin{1//2}; `::Spin{1//2}` is the type dispatch argument (no variable name needed because the argument is not used in the body)
    Sz = ComplexF64[1 0; 0 -1] / 2          # ½·diag(+1,−1); Sz|↑⟩=+½|↑⟩  # `ComplexF64[...]` is a 2×2 complex matrix literal 
    Sp = ComplexF64[0 1; 0 0]              # S⁺|↓⟩=|↑⟩, S⁺|↑⟩=0  # S⁺ raises the spin: |↓⟩→|↑⟩; upper-triangular entry [1,2] = 1
    Sm = Sp'                                 # S⁻=(S⁺)†  # `A'` = conjugate transpose ; S⁻ lowers the spin
    Sx = (Sp + Sm) / 2   # Sx = ½(S⁺ + S⁻); the physical transverse spin operator
    Sy = (Sp - Sm) / (2im)   # `2im` is the imaginary number 2i in Julia. Sy = (S⁺ - S⁻)/(2i)
    I2 = LinearAlgebra.one(Sz)   # `LinearAlgebra.one(M)` = identity matrix of same size/type as M ; `one` is Julia's generic "multiplicative identity" function
    (; Sx, Sy, Sz, Sp, Sm, I=I2)   # `(; ...)` creates a NamedTuple 
end

# todo: switch to using a simple function that calculates the required matrix elements for a given S using wigner-eckart theorem instead of hard-coding separately for 1//2 and 1.
function algebra_generators(::Spin{1})   # method for spin-1; separate dispatch from Spin{1//2} even though the formula is similar — Julia selects this at compile time with zero overhead
    # 3×3 spin-1 matrices (Condon–Shortley convention).
    # Basis ordering: |+1⟩, |0⟩, |−1⟩  (mz = 1, 0, −1).
    Sz = ComplexF64[1 0 0; 0 0 0; 0 0 -1]   # 3×3 diagonal matrix: Sz = diag(1, 0, −1) for mz=+1, 0, −1
    Sp = ComplexF64[0 √2 0; 0 0 √2; 0 0 0]   # S⁺: raises mz by 1  # `√2` = `sqrt(2)` (Julia allows Unicode math symbols); S⁺|mz=0⟩=√2|mz=1⟩, S⁺|mz=-1⟩=√2|mz=0⟩; from the Condon-Shortley formula sqrt(S(S+1)−mz(mz+1))
    Sm = Sp'   # S⁻ = (S⁺)†
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / (2im)
    I3 = LinearAlgebra.one(Sz)   # 3×3 identity
    (; Sx, Sy, Sz, Sp, Sm, I=I3)   # NamedTuple with all spin-1 operators
end

function algebra_generators(::SpinlessFermion)
    # Basis ordering: |0⟩ (vacuum, index 1), |1⟩ (occupied, index 2).
    # c destroys a particle: c|1⟩ = |0⟩, c|0⟩ = 0.
    c = ComplexF64[0 1; 0 0]   # annihilation operator: c[row,col] = ⟨row|c|col⟩; c[1,2]=1 means c|1⟩=|0⟩; all other entries are 0
    cdag = c'   # creation operator c† = c^T (real matrix, so just transpose); `c'` = conjugate transpose
    I2 = LinearAlgebra.one(c)   # 2×2 identity
    n = cdag * c   # number operator: diag(0,1)  # n = c†c; explicitly computed (not hardcoded) to ensure consistency with c; `*` is matrix multiplication
    (; c, cdag, n, I=I2)   # NamedTuple; `I=I2` renames I2 as `I`
end

function algebra_generators(::HardCoreBoson)
    # Identical matrix structure to SpinlessFermion, but commuting statistics.
    # Basis ordering: |0⟩ (vacuum, index 1), |1⟩ (occupied, index 2).
    b = ComplexF64[0 1; 0 0]   # bosonic annihilator; same matrix as SpinlessFermion's `c` but field name `b` signals bosonic statistics to the caller
    bdag = b'   # creation operator b†
    n = bdag * b   # occupation number operator
    I2 = LinearAlgebra.one(b)   # 2×2 identity
    (; b, bdag, n, I=I2)   # NamedTuple; note different field names from SpinlessFermion (b/bdag vs c/cdag)
end

function algebra_generators(::Electron)
    # 4×4 matrices on the electron site.
    # Basis ordering: {|0⟩, |↑⟩, |↓⟩, |↑↓⟩} — "spin-up first" convention
    # (ITensor / Essler et al.).  |↑↓⟩ ≡ c†↑ c†↓ |0⟩, so c↓|↑↓⟩ = −|↑⟩.

    # c↑: destroys up-spin.  |↑⟩→|0⟩  and  |↑↓⟩→|↓⟩  (no sign, up acts first).
    cup = ComplexF64[   # 4×4 annihilator for spin-up; cup[1,2]=1 means ⟨0|c↑|↑⟩=1; cup[3,4]=1 means ⟨↓|c↑|↑↓⟩=1 (removing ↑ from a doubly-occupied site leaves ↓)
        0 1 0 0;   # row 1 (|0⟩): cup destroys ↑ from |↑⟩ to give |0⟩
        0 0 0 0;   # row 2 (|↑⟩): no ↑ to destroy here
        0 0 0 1;   # row 3 (|↓⟩): cup destroys ↑ from |↑↓⟩ to give |↓⟩
        0 0 0 0    # row 4 (|↑↓⟩): no further ↑ transition
    ]

    # c↓: destroys down-spin.  |↓⟩→|0⟩  and  |↑↓⟩→−|↑⟩  (−1 from ordering).
    cdn = ComplexF64[   # 4×4 annihilator for spin-down; cdn[1,3]=1 means ⟨0|c↓|↓⟩=1; cdn[2,4]=-1 means ⟨↑|c↓|↑↓⟩=−1 (anticommuting past the already-present ↑ gives a sign)
        0 0 1 0;   # row 1 (|0⟩): cdn destroys ↓ from |↓⟩ to give |0⟩
        0 0 0 -1;  # row 2 (|↑⟩): cdn destroys ↓ from |↑↓⟩ to give −|↑⟩ (−1 from fermionic anticommutation)
        0 0 0 0;   # row 3 (|↓⟩): no further ↓ transition
        0 0 0 0    # row 4 (|↑↓⟩): no further ↓ transition
    ]

    cupdag = cup'   # c†↑ = (c↑)†; spin-up creation operator
    cdndag = cdn'   # c†↓ = (c↓)†; spin-down creation operator
    nup = cupdag * cup   # n↑ = c†↑c↑; spin-up number operator; diag(0,1,0,1) in the {|0⟩,|↑⟩,|↓⟩,|↑↓⟩} basis
    ndn = cdndag * cdn   # n↓ = c†↓c↓; spin-down number operator; diag(0,0,1,1)
    I4 = LinearAlgebra.one(cup)   # 4×4 identity
    n = nup + ndn   # total occupation number: n = n↑ + n↓; diag(0,1,1,2)

    # Spin operators built from the electron operators (for observables)
    Sz = (nup - ndn) / 2   # Sz = (n↑ − n↓)/2; spin of the electron
    Sp = cupdag * cdn    # S⁺ = c†↑ c↓  # S⁺ flips spin ↓→↑: S⁺|↓⟩=|↑⟩; S⁺|0⟩=S⁺|↑↓⟩=0
    Sm = Sp'   # S⁻ = (S⁺)†

    (; cup, cdn, cupdag, cdndag, nup, ndn, n, Sz, Sp, Sm, I=I4)   # NamedTuple with all electron operators
end

function algebra_generators(::MajoranaFermion)
    # MajoranaFermion operators on the paired-fermion site.
    # γ₁ = c + c†  (=σˣ on the Fock site),  γ₂ = i(c† − c)  (=σʸ).
    # Both are Hermitian: γ†=γ.  Algebra: {γₐ,γᵦ}=2δₐᵦ.
    ops = algebra_generators(SpinlessFermion())   # reuse the spinless-fermion operators; DRY principle: Majorana operators are linear combinations of c and c†
    γ1 = ops.c + ops.cdag   # γ₁ = c + c†; Hermitian: γ₁† = c† + c = γ₁; equals σˣ in the {|0⟩,|1⟩} Fock basis
    γ2 = im * (ops.cdag - ops.c)   # `im` = the imaginary unit. γ₂ = i(c† − c); Hermitian: γ₂† = (−i)(c − c†) = i(c† − c) = γ₂; equals σʸ in the Fock basis
    I2 = LinearAlgebra.one(γ1)   # 2×2 identity
    (; γ1, γ2, I=I2)   # NamedTuple; `γ1` and `γ2` use Unicode Greek letters (Julia allows Unicode identifiers — Python 3 does too)
end

"""
    canonical_relation(dof::AbstractDoF) -> CanonicalRelation

Return the intrinsic inter-site statistics of `dof` as either a [`CCR`](@ref)
or [`CAR`](@ref) instance.

This drives two downstream decisions:

 1. **Jordan–Wigner string insertion** — when computing expectation values or
    building MPOs for non-adjacent fermionic operators, a string factor
    ``\\prod_{k=i}^{j-1} (-1)^{n_k}`` must be inserted between sites ``i`` and ``j``.
    Code that needs this check calls `canonical_relation(dof) isa CAR`.

 2. **Native graded spaces** (Week 12) — when TensorKit integration is enabled,
    `CAR` DoFs will use graded vector spaces and fuse/split operations
    that handle signs automatically.

| DoF               | `canonical_relation` |
|:----------------- |:-------------------- |
| `Spin{S}`         | `CCR()`              |
| `HardCoreBoson`   | `CCR()`              |
| `SpinlessFermion` | `CAR()`              |
| `Electron`        | `CAR()`              |
| `MajoranaFermion` | `CAR()`              |

# Examples

```jldoctest
julia> canonical_relation(SpinHalf()) isa CCR
true

julia> canonical_relation(SpinlessFermion()) isa CAR
true

julia> canonical_relation(HardCoreBoson()) isa CCR
true
```
"""
canonical_relation(::Spin) = CCR()   # all Spin{S} types → bosonic statistics; `::Spin` matches any Spin{S} regardless of S (more general than `::Spin{1//2}`); `CCR()` constructs the CCR singleton instance
canonical_relation(::HardCoreBoson) = CCR()   # hard-core bosons commute even though their matrix structure mirrors fermions
canonical_relation(::SpinlessFermion) = CAR()   # `CAR()` constructs the CAR singleton; fermionic DoFs anticommute on different sites
canonical_relation(::Electron) = CAR()   # both spin-up and spin-down electrons are fermionic
canonical_relation(::MajoranaFermion) = CAR()   # Majorana fermions inherit CAR statistics from the underlying complex fermion
