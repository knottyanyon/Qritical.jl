using HalfIntegers
using TensorKit: TensorKit

# ── AbstractSite ──────────────────────────────────────────────────────────────

abstract type AbstractSite end

# ── SpinSite ──────────────────────────────────────────────────────────────────

struct SpinSite <: AbstractSite
    spin_quantum_number::HalfInt
    lattice_ordinal::Int
    space::TensorKit.ElementarySpace
end

function SpinSite(
    spin_quantum_number::HalfInt, lattice_ordinal::Int; symmetry::Symbol=:none
)
    ## validation
    spin_quantum_number >= 0 || throw(
        ArgumentError("spin_quantum_number must be non-negative, got $spin_quantum_number"),
    )

    ## symmetry
    space = if symmetry == :none
        # ℂⁿ — just an n-dimensional complex vector space with no sectors, no block structure
        TensorKit.ComplexSpace(Int(2 * spin_quantum_number + 1))
    elseif symmetry == :U1
        # sectors: Sz = +S, S-1, ..., -S each with multiplicity 1
        pairs = [m => 1 for m in spin_quantum_number:-1:(-spin_quantum_number)]
        TensorKit.U1Space(pairs...)
    elseif symmetry == :SU2
        # one sector labeled by total spin j = S, multiplicity 1
        TensorKit.SU2Space(spin_quantum_number => 1)
    else
        throw(ArgumentError("SpinSite does not support symmetry=:$symmetry. Use :U1 or :SU2"))
    end
    return SpinSite(spin_quantum_number, lattice_ordinal, space)
end

# ── SpinlessFermionicSite ─────────────────────────────────────────────────────

struct SpinlessFermionicSite <: AbstractSite
    lattice_ordinal::Int
    space::TensorKit.ElementarySpace
end

function SpinlessFermionicSite(lattice_ordinal::Int; symmetry::Symbol=:none)
    space = if symmetry == :none
        TensorKit.ComplexSpace(2)
    elseif symmetry == :U1
        TensorKit.U1Space(0 => 1, 1 => 1)   # n=0 empty, n=1 occupied
    elseif symmetry == :Z2
        TensorKit.Z2Space(0 => 1, 1 => 1)   # even/odd parity
    else
        throw(
            ArgumentError(
                "SpinlessFermionicSite does not support symmetry=:$symmetry. Use :U1, :Z2, or :none",
            ),
        )
    end
    return SpinlessFermionicSite(lattice_ordinal, space)
end

# ── SpinlessHardCoreBosonicSite ───────────────────────────────────────────────

struct SpinlessHardCoreBosonicSite <: AbstractSite
    lattice_ordinal::Int
    space::TensorKit.ElementarySpace
end

function SpinlessHardCoreBosonicSite(lattice_ordinal::Int; symmetry::Symbol=:none)
    space = if symmetry == :none
        TensorKit.ComplexSpace(2)
    elseif symmetry == :U1
        TensorKit.U1Space(0 => 1, 1 => 1)   # n=0 empty, n=1 occupied
    else
        throw(
            ArgumentError(
                "SpinlessHardCoreBosonicSite does not support symmetry=:$symmetry. Use :U1, or :none",
            ),
        )
    end
    return SpinlessHardCoreBosonicSite(lattice_ordinal, space)
end

# ── SpinlessBosonicSite ───────────────────────────────────────────────────────

struct SpinlessBosonicSite <: AbstractSite
    lattice_ordinal::Int
    n_max_occ::Int # occupation cutoff
    space::TensorKit.ElementarySpace
end

function SpinlessBosonicSite(lattice_ordinal::Int; n_max_occ::Int, symmetry::Symbol=:none)
    ## validation
    n_max_occ >= 0 || throw(ArgumentError("n_max_occ must be non-negative, got $n_max_occ"))
    # build space for :none, :U1
    space = if symmetry == :none
        TensorKit.ComplexSpace(n_max_occ + 1)
    elseif symmetry == :U1
        pairs = [m => 1 for m in 0:n_max_occ]
        TensorKit.U1Space(pairs...)

    else
        throw(
            ArgumentError(
                "SpinlessBosonicSite does not support symmetry=:$symmetry. Use :U1, or :none",
            ),
        )
    end

    return SpinlessBosonicSite(lattice_ordinal, n_max_occ, space)
end

local_hilbert_dim(s::SpinSite) = Int(2 * s.spin_quantum_number + 1)
local_hilbert_dim(::SpinlessFermionicSite) = 2
local_hilbert_dim(::SpinlessHardCoreBosonicSite) = 2
local_hilbert_dim(s::SpinlessBosonicSite) = s.n_max_occ + 1