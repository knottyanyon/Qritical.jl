# §3 / §6.1  Degrees-of-freedom layer.
#
# A DoF is the physical content placed at each lattice site: its local Hilbert
# space, operator algebra, and intrinsic inter-site statistics.  The statistics
# determines how inter-site signs are handled (Jordan–Wigner string vs. native
# fermionic grading).

# ─────────────────────────────────────────────────────────────────────────────
# Abstract DoF supertype
# ─────────────────────────────────────────────────────────────────────────────

abstract type AbstractDoF end

# ─────────────────────────────────────────────────────────────────────────────
# Concrete DoFs
# ─────────────────────────────────────────────────────────────────────────────

struct Spin{S} <: AbstractDoF end       # S = 1//2, 1, 3//2, …; local_dim = 2S+1
const SpinHalf = Spin{1//2}
const SpinOne  = Spin{1}

struct SpinlessFermion <: AbstractDoF end   # 2D site {|0⟩,|1⟩}; c, c†, n
struct Electron        <: AbstractDoF end   # 4D site {|0⟩,|↑⟩,|↓⟩,|↑↓⟩}
struct Majorana        <: AbstractDoF end   # Majorana modes on a paired fermion site
struct HardCoreBoson   <: AbstractDoF end   # 2D site {|0⟩,|1⟩}; b, b†, n; b²=0

# ─────────────────────────────────────────────────────────────────────────────
# Statistics — intrinsic inter-site statistics of the DoF
# ─────────────────────────────────────────────────────────────────────────────

abstract type Statistics end
struct Commuting     <: Statistics end   # operators on different sites commute
struct Anticommuting <: Statistics end   # operators on different sites anticommute

statistics(::Spin)            = Commuting()
statistics(::HardCoreBoson)   = Commuting()
statistics(::SpinlessFermion) = Anticommuting()
statistics(::Electron)        = Anticommuting()
statistics(::Majorana)        = Anticommuting()

# ─────────────────────────────────────────────────────────────────────────────
# local_dim — local Hilbert-space dimension
# ─────────────────────────────────────────────────────────────────────────────

local_dim(::Spin{S}) where {S} = Int(2S + 1)   # computed at compile time
local_dim(::SpinlessFermion)   = 2
local_dim(::Electron)          = 4
local_dim(::HardCoreBoson)     = 2
local_dim(::Majorana)          = 2   # per paired (complex-fermion) site

# ─────────────────────────────────────────────────────────────────────────────
# Symmetry tags — sectorless for now; Week 12 upgrades to graded spaces
# ─────────────────────────────────────────────────────────────────────────────

struct NoSymmetry end

physical_space(dof::AbstractDoF, ::NoSymmetry) = local_dim(dof)

# ─────────────────────────────────────────────────────────────────────────────
# operators — on-site operator matrices as a NamedTuple
# ─────────────────────────────────────────────────────────────────────────────

function operators(::Spin{1//2})
    I2 = ComplexF64[1 0; 0 1]
    Sz = ComplexF64[1 0; 0 -1] / 2          # ½·diag(+1,−1); Sz|↑⟩=+½|↑⟩
    Sp = ComplexF64[0 1; 0  0]              # S⁺|↓⟩=|↑⟩, S⁺|↑⟩=0
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
    c    = ComplexF64[0 1; 0 0]
    cdag = c'
    n    = cdag * c   # number operator: diag(0,1)
    (; c, cdag, n, I=I2)
end

function operators(::HardCoreBoson)
    # Identical matrix structure to SpinlessFermion, but commuting statistics.
    # Basis ordering: |0⟩ (vacuum, index 1), |1⟩ (occupied, index 2).
    I2   = ComplexF64[1 0; 0 1]
    b    = ComplexF64[0 1; 0 0]
    bdag = b'
    n    = bdag * b
    (; b, bdag, n, I=I2)
end

function operators(::Electron)
    # 4×4 matrices on the electron site.
    # Basis ordering: {|0⟩, |↑⟩, |↓⟩, |↑↓⟩} — "spin-up first" convention
    # (ITensor / Essler et al.).  |↑↓⟩ ≡ c†↑ c†↓ |0⟩, so c↓|↑↓⟩ = −|↑⟩.
    I4 = ComplexF64(1) * I(4)

    # c↑: destroys up-spin.  |↑⟩→|0⟩  and  |↑↓⟩→|↓⟩  (no sign, up acts first).
    cup = ComplexF64[0 1 0 0;
                     0 0 0 0;
                     0 0 0 1;
                     0 0 0 0]

    # c↓: destroys down-spin.  |↓⟩→|0⟩  and  |↑↓⟩→−|↑⟩  (−1 from ordering).
    cdn = ComplexF64[0 0 1  0;
                     0 0 0 -1;
                     0 0 0  0;
                     0 0 0  0]

    cupdag = cup'
    cdndag = cdn'
    nup    = cupdag * cup
    ndn    = cdndag * cdn
    n      = nup + ndn

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
    I2  = ComplexF64[1 0; 0 1]
    γ1  = ops.c + ops.cdag
    γ2  = im * (ops.cdag - ops.c)
    (; γ1, γ2, I=I2)
end
