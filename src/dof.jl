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

 2. **Operator algebra** — the set of elementary matrices that act on that space, returned as a `NamedTuple` by [`operators`](@ref).

 3. **Exchange statistics** — whether operators on different sites
    commute ([`Commuting`](@ref)) or anticommute ([`Anticommuting`](@ref)),
    as returned by [`statistics`](@ref).  This drives the choice between
    Jordan–Wigner string insertion and native fermionic grading.

Every concrete DoF in Qritical.jl is a subtype of `AbstractDoF` .

Concrete subtypes: [`Spin`](@ref), [`SpinlessFermion`](@ref), [`Electron`](@ref),
[`Majorana`](@ref), [`HardCoreBoson`](@ref).

See also: [`local_dim`](@ref), [`statistics`](@ref), [`operators`](@ref),
[`physical_space`](@ref)
"""
abstract type AbstractDoF end

# ----------------------------------------------------------------------------------------
# Concrete DoFs
# ----------------------------------------------------------------------------------------

"""
    Spin{S} <: AbstractDoF

A spin-``S`` degree of freedom site with local Hilbert-space dimension ``2S+1``.

The type parameter `S` is a `Rational{Int}` such as `1//2`, `1`, `3//2`, etc. The local basis is ordered by descending magnetic quantum number: ``|m_z = S\\rangle, |m_z = S-1\\rangle, \\ldots, |m_z = -S\\rangle``.

Operators are built using the **Condon–Shortley phase convention** (same as most quantum chemistry and condensed-matter textbooks), so the ladder operators satisfy

```math
S^{\\pm} |m\\rangle = \\sqrt{S(S+1) - m(m \\pm 1)} \\, |m \\pm 1\\rangle.
```

Use the pre-defined aliases [`SpinHalf`](@ref) and [`SpinOne`](@ref) for the
most common cases rather than writing `Spin{1//2}()` or `Spin{1}()` by hand.

# Examples

```jldoctest
julia> local_dim(Spin{3//2}())
4

julia> local_dim(SpinHalf())
2
```
"""
struct Spin{S} <: AbstractDoF end       # S = 1//2, 1, 3//2, …; local_dim = 2S+1

"""
    SpinHalf

Alias for `Spin{1//2}` — a two-level spin site with local Hilbert
space ``\\{|{\\uparrow}\\rangle, |{\\downarrow}\\rangle\\}`` and ``d = 2``.

# Examples

```jldoctest
julia> local_dim(SpinHalf())
2

julia> statistics(SpinHalf()) isa Commuting
true
```
"""
const SpinHalf = Spin{1//2}

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
const SpinOne = Spin{1}

"""
    SpinlessFermion <: AbstractDoF

A spinless (single-orbital) fermionic site with local dimension ``d = 2``.

The local Hilbert space is ``\\{|0\\rangle, |1\\rangle\\}`` — vacuum and a single
occupied state.  The elementary operators are the annihilation operator ``c``
(destroys the particle, ``c|1\\rangle = |0\\rangle``) and the number operator
``n = c^\\dagger c`` with eigenvalues 0 and 1.

Because this is a fermionic site, [`statistics`](@ref) returns [`Anticommuting`](@ref),
meaning that operators on different sites anticommute:
``\\{c_i, c_j^\\dagger\\} = \\delta_{ij}``.
When working without native graded spaces, Jordan–Wigner string factors must be
inserted for non-adjacent operators.

See also: [`HardCoreBoson`](@ref) (same matrix structure, commuting statistics),
[`Electron`](@ref) (spin-½ version), [`operators`](@ref)
"""
struct SpinlessFermion <: AbstractDoF end   # 2D site {|0⟩,|1⟩}; c, c†, n

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

Inter-site statistics are [`Anticommuting`](@ref) — both ``c_{\\uparrow}`` and
``c_{\\downarrow}`` are fermionic.

The full operator set (see [`operators(::Electron)`](@ref)) includes individual
spin-resolved annihilators `cup`/`cdn`, number operators `nup`/`ndn`, the total
number `n`, and spin operators `Sz`, `Sp`, `Sm`.

See also: [`SpinlessFermion`](@ref), [`operators`](@ref)
"""
struct Electron <: AbstractDoF end   # 4D site {|0⟩,|↑⟩,|↓⟩,|↑↓⟩} 

"""
    Majorana <: AbstractDoF

A Majorana fermion site, realized on a paired (single complex-fermion) site with
local dimension ``d = 2``.

A Majorana operator ``\\gamma`` is its own Hermitian conjugate (``\\gamma^\\dagger = \\gamma``)
and satisfies the algebra ``\\{\\gamma_a, \\gamma_b\\} = 2\\delta_{ab}``.  In
Qritical.jl two Majorana modes are defined on each site via the decomposition of
a complex fermion ``c`` into real and imaginary parts:

```math
\\gamma_1 = c + c^\\dagger = \\sigma^x, \\qquad
\\gamma_2 = i(c^\\dagger - c) = \\sigma^y.
```

Both ``\\gamma_1`` and ``\\gamma_2`` are Hermitian, traceless, and square to the
identity on the local two-dimensional space.  The inter-site statistics are
[`Anticommuting`](@ref), inherited from the underlying complex fermion.

See also: [`operators`](@ref)
"""
struct Majorana <: AbstractDoF end   # Majorana modes on a paired fermion site

"""
    HardCoreBoson <: AbstractDoF

A hard-core bosonic site with local dimension ``d = 2``.

Hard-core bosons obey the constraint ``b^2 = 0`` (at most one boson per site),
which gives a two-dimensional local Hilbert space ``\\{|0\\rangle, |1\\rangle\\}``.
Algebraically this looks identical to a spinless fermion site, but the crucial
difference is that bosonic operators on different sites **commute**:
``[b_i, b_j^\\dagger] = 0`` for ``i \\neq j``.

This means [`statistics`](@ref) returns [`Commuting`](@ref) and no Jordan–Wigner
strings are needed for non-adjacent operators.

See also: [`SpinlessFermion`](@ref) (anticommuting version), [`operators`](@ref)
"""
struct HardCoreBoson <: AbstractDoF end   # 2D site {|0⟩,|1⟩}; b, b†, n; b²=0

# ----------------------------------------------------------------------------------------
# Statistics — intrinsic inter-site statistics of the DoF
# ----------------------------------------------------------------------------------------

"""
    Statistics

Abstract supertype for the intrinsic inter-site statistics of a [`AbstractDoF`](@ref).

Statistics determines how operators on different sites combine:

  - [`Commuting`](@ref): ``[O_i, O_j] = 0`` for ``i \\neq j``.  No sign factors needed.
  - [`Anticommuting`](@ref): ``\\{O_i, O_j\\} = 0`` for ``i \\neq j``.  Signs must be
    tracked, either via Jordan–Wigner strings or via native fermionic grading.

Use [`statistics`](@ref) to query the statistics of any concrete `AbstractDoF`.
"""

abstract type Statistics end

"""
    Commuting <: Statistics

Tag indicating that operators on different sites commute: ``[O_i, O_j] = 0``
for ``i \\neq j``.

Assigned to bosonic DoFs: [`Spin`](@ref) (all values of ``S``) and
[`HardCoreBoson`](@ref).  No Jordan–Wigner string is ever needed between
commuting sites.

See also: [`Anticommuting`](@ref), [`statistics`](@ref)
"""
struct Commuting <: Statistics end   # operators on different sites commute

"""
    Anticommuting <: Statistics

Tag indicating that operators on different sites anticommute: ``\\{O_i, O_j\\} = 0``
for ``i \\neq j``.

Assigned to fermionic DoFs: [`SpinlessFermion`](@ref), [`Electron`](@ref), and
[`Majorana`](@ref).  When computing expectation values or building MPOs for
non-adjacent operators, a Jordan–Wigner string ``\\prod_k (-1)^{n_k}`` must be
inserted between sites ``i`` and ``j``.  In the future this will be
replaced by native graded-space arithmetic via TensorKit.

See also: [`Commuting`](@ref), [`statistics`](@ref)
"""
struct Anticommuting <: Statistics end   # operators on different sites anticommute

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
| `Majorana`        | 2           |
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
local_dim(::Spin{S}) where {S} = Int(2S + 1)   # computed at compile time
local_dim(::SpinlessFermion) = 2
local_dim(::Electron) = 4
local_dim(::HardCoreBoson) = 2
local_dim(::Majorana) = 2   # per paired (complex-fermion) site

# ----------------------------------------------------------------------------------------
# Symmetry tags — sectorless for now; future upgrades to graded spaces
# ----------------------------------------------------------------------------------------

"""
    NoSymmetry

Tag type selecting the sectorless (dense, symmetry-ignorant) backend for tensor
and operator construction.

Right now every physical space in Qritical.jl uses this backend: operators are
plain `ComplexF64` matrices and MPS bond spaces are ungraded `Int` dimensions.

In future this will be replaced by a graded `ElementarySpace` from TensorKit.jl
that carries quantum-number sectors (e.g. ``U(1)`` particle number or
``SU(2)`` spin).  The tag is introduced now so that every function signature
that will eventually dispatch on symmetry already has a slot for it; switching
backends will then only require adding new methods.

See also: [`physical_space`](@ref)
"""
struct NoSymmetry end

"""
    physical_space(dof::AbstractDoF, ::NoSymmetry) -> Int

Return an integer representing the local physical space of `dof` in the
sectorless backend.

Currently this simply returns [`local_dim(dof)`](@ref) — a plain `Int` that
the MPS/MPO constructors use as the physical-leg dimension.

When symmetry support is added in the future, this function will
gain new methods dispatching on symmetry tags such as `U1Symmetry()` and will
return a graded `ElementarySpace` from TensorKit.jl carrying the full
quantum-number sector structure.  All Hamiltonian and MPS construction code
should call `physical_space` rather than `local_dim` so it automatically
benefits from that upgrade.

# Examples

```jldoctest
julia> physical_space(SpinHalf(), NoSymmetry())
2

julia> physical_space(Electron(), NoSymmetry())
4
```
"""

physical_space(dof::AbstractDoF, ::NoSymmetry) = local_dim(dof)

# ----------------------------------------------------------------------------------------
# operators — on-site operator matrices as a NamedTuple
# ----------------------------------------------------------------------------------------

"""
    operators(dof::AbstractDoF) -> NamedTuple

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
field names `b`, `bdag`, `n`, `I`.  Statistics are [`Commuting`](@ref).

**`Electron`** — ``d = 4``, basis ``\\{|0\\rangle, |{\\uparrow}\\rangle, |{\\downarrow}\\rangle, |{\\uparrow\\downarrow}\\rangle\\}``
("spin-up first" convention):

  - `cup`, `cupdag` — spin-up annihilator/creator
  - `cdn`, `cdndag` — spin-down annihilator/creator; note ``c_{\\downarrow}|{\\uparrow\\downarrow}\\rangle = -|{\\uparrow}\\rangle``
  - `nup`, `ndn`, `n` — partial and total number operators
  - `Sz`, `Sp`, `Sm` — spin operators built from ``c^\\dagger_\\uparrow c_\\downarrow`` etc.
  - `I`  — ``4\\times 4`` identity

**`Majorana`** — ``d = 2``, two Hermitian Majorana operators:

  - `γ1 = c + c†` (equals ``\\sigma^x`` on the Fock site)
  - `γ2 = i(c† - c)` (equals ``\\sigma^y``)
  - `I`  — identity

# Examples

```jldoctest
julia> ops = operators(SpinHalf());

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
function operators(::Spin{1//2})
    I2 = ComplexF64[1 0; 0 1]
    Sz = ComplexF64[1 0; 0 -1] / 2          # ½·diag(+1,−1); Sz|↑⟩=+½|↑⟩
    Sp = ComplexF64[0 1; 0 0]              # S⁺|↓⟩=|↑⟩, S⁺|↑⟩=0
    Sm = Sp'                                 # S⁻=(S⁺)†
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / (2im)
    (; Sx, Sy, Sz, Sp, Sm, I=I2)
end

function operators(::Spin{1})
    # 3×3 spin-1 matrices (Condon–Shortley convention).
    # Basis ordering: |+1⟩, |0⟩, |−1⟩  (mz = 1, 0, −1).
    I3 = ComplexF64[1 0 0; 0 1 0; 0 0 1]
    Sz = ComplexF64[1 0 0; 0 0 0; 0 0 -1]
    Sp = ComplexF64[0 √2 0; 0 0 √2; 0 0 0]   # S⁺: raises mz by 1
    Sm = Sp'
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / (2im)
    (; Sx, Sy, Sz, Sp, Sm, I=I3)
end

function operators(::SpinlessFermion)
    I2 = ComplexF64[1 0; 0 1]
    # Basis ordering: |0⟩ (vacuum, index 1), |1⟩ (occupied, index 2).
    # c destroys a particle: c|1⟩ = |0⟩, c|0⟩ = 0.
    c = ComplexF64[0 1; 0 0]
    cdag = c'
    n = cdag * c   # number operator: diag(0,1)
    (; c, cdag, n, I=I2)
end

function operators(::HardCoreBoson)
    # Identical matrix structure to SpinlessFermion, but commuting statistics.
    # Basis ordering: |0⟩ (vacuum, index 1), |1⟩ (occupied, index 2).
    I2 = ComplexF64[1 0; 0 1]
    b = ComplexF64[0 1; 0 0]
    bdag = b'
    n = bdag * b
    (; b, bdag, n, I=I2)
end

function operators(::Electron)
    # 4×4 matrices on the electron site.
    # Basis ordering: {|0⟩, |↑⟩, |↓⟩, |↑↓⟩} — "spin-up first" convention
    # (ITensor / Essler et al.).  |↑↓⟩ ≡ c†↑ c†↓ |0⟩, so c↓|↑↓⟩ = −|↑⟩.
    I4 = ComplexF64(1) * I(4)

    # c↑: destroys up-spin.  |↑⟩→|0⟩  and  |↑↓⟩→|↓⟩  (no sign, up acts first).
    cup = ComplexF64[
        0 1 0 0;
        0 0 0 0;
        0 0 0 1;
        0 0 0 0
    ]

    # c↓: destroys down-spin.  |↓⟩→|0⟩  and  |↑↓⟩→−|↑⟩  (−1 from ordering).
    cdn = ComplexF64[
        0 0 1 0;
        0 0 0 -1;
        0 0 0 0;
        0 0 0 0
    ]

    cupdag = cup'
    cdndag = cdn'
    nup = cupdag * cup
    ndn = cdndag * cdn
    n = nup + ndn

    # Spin operators built from the electron operators (for observables)
    Sz = (nup - ndn) / 2
    Sp = cupdag * cdn    # S⁺ = c†↑ c↓
    Sm = Sp'

    (; cup, cdn, cupdag, cdndag, nup, ndn, n, Sz, Sp, Sm, I=I4)
end

function operators(::Majorana)
    # Majorana operators on the paired-fermion site.
    # γ₁ = c + c†  (=σˣ on the Fock site),  γ₂ = i(c† − c)  (=σʸ).
    # Both are Hermitian: γ†=γ.  Algebra: {γₐ,γᵦ}=2δₐᵦ.
    ops = operators(SpinlessFermion())
    I2 = ComplexF64[1 0; 0 1]
    γ1 = ops.c + ops.cdag
    γ2 = im * (ops.cdag - ops.c)
    (; γ1, γ2, I=I2)
end

"""
    statistics(dof::AbstractDoF) -> Statistics

Return the intrinsic inter-site statistics of `dof` as either a [`Commuting`](@ref)
or [`Anticommuting`](@ref) instance.

This drives two downstream decisions:

 1. **Jordan–Wigner string insertion** — when computing expectation values or
    building MPOs for non-adjacent fermionic operators, a string factor
    ``\\prod_{k=i}^{j-1} (-1)^{n_k}`` must be inserted between sites ``i`` and ``j``.
    Code that needs this check calls `statistics(dof) isa Anticommuting`.

 2. **Native graded spaces** (Week 12) — when TensorKit integration is enabled,
    `Anticommuting` DoFs will use graded vector spaces and fuse/split operations
    that handle signs automatically.

| DoF               | `statistics` result |
|:----------------- |:------------------- |
| `Spin{S}`         | `Commuting()`       |
| `HardCoreBoson`   | `Commuting()`       |
| `SpinlessFermion` | `Anticommuting()`   |
| `Electron`        | `Anticommuting()`   |
| `Majorana`        | `Anticommuting()`   |

# Examples

```jldoctest
julia> statistics(SpinHalf()) isa Commuting
true

julia> statistics(SpinlessFermion()) isa Anticommuting
true

julia> statistics(HardCoreBoson()) isa Commuting
true
```
"""
statistics(::Spin) = Commuting()
statistics(::HardCoreBoson) = Commuting()
statistics(::SpinlessFermion) = Anticommuting()
statistics(::Electron) = Anticommuting()
statistics(::Majorana) = Anticommuting()
