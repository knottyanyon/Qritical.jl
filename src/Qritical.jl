module Qritical

using LinearAlgebra
using TensorOperations
import SparseArrays
import SparseArrays: sparse

"""
    AbstractGeometry

Abstract supertype for lattice geometries.

A geometry answers exactly two queries: which sites exist, and which pairs of
sites are connected by bonds.  That is all the Hamiltonian builder needs — it
does not care whether the underlying graph is a chain, a square lattice, or a
torus.  Concrete subtypes only need to implement [`sites`](@ref) and
[`bonds`](@ref).

Current concrete geometry: [`Chain`](@ref) (1D open/periodic chain).
Planned extensions: `Square`, `Torus`, `Lattice{V,E}` (see §2 of the design plan).

See also: [`Chain`](@ref), [`sites`](@ref), [`bonds`](@ref)
"""
abstract type AbstractGeometry end

include("geometry.jl")
include("dof.jl")
include("symmetries.jl")
include("operator.jl")
include("indices.jl")
include("qtensor.jl")
include("spectrum.jl")
include("svd.jl")
include("io.jl")
include("mps.jl")
include("canonicalize.jl")
include("vidal.jl")
include("correlators.jl")
include("finite_mpo.jl")
include("power_method.jl")
include("tebd.jl")
include("quench.jl")
include("ed.jl")
include("disorder.jl")

# ==== Index layer =============================================================
export IxLoc, Upper, Lower
export AbstractIx, TIx, MulTIx
export dim, label, which_space
export upper, lower, uppers, lowers, uppers_range, lowers_range, bond_label

# ==== QTensor + partitions ====================================================
export QTensor
export Partition, Bipartition, complement, bipartition, group_legs

# ==== SVD + truncation ========================================================
export AbstractTrunc, NoTrunc, MaxBondDimTrunc, ValCutoffTrunc
export FullSVD, ReducedSVD, do_svd

# ==== Spectrum + orthogonality centre =========================================
export Bond, OrthoCenter, BondCenter, SiteCenter
export AbstractSpectrum, SingValSpectrum, EigValSpectrum, SchmidtSpectrum
export schmidt_rank, spectral_gap, schmidt_values
export entanglement_entropy, entanglement_spectrum

# ==== State utilities + I/O ===================================================
export bipartition_matrix, as_state, load_array

# ==== MPS & canonical forms ==================================================
export AbstractMPSForm, CanonicalForm, VidalForm, ArbitraryForm
export FiniteMPS, to_mps, add_mps
export CanonicalizeConfig, LeftCanonical, RightCanonical, BondCanonical, SiteCanonical
export canonicalize, canonical_error, is_canonical, overlap, local_expectation, two_point
export to_vidal, to_canonical

# ==== Geometry ================================================================
export AbstractGeometry, Chain, sites, bonds

# ==== DoF layer ===============================================================
export AbstractDoF
export Spin, SpinHalf, SpinOne
export SpinlessFermion, Electron, MajoranaFermion, HardCoreBoson
export CanonicalRelation, CCR, CAR
export local_dim, canonical_relation, algebra_generators
# ==== Symmetry tags ===========================================================
export NoSymmetry, physical_space

# ==== Operator / Hamiltonian ==================================================
export uniform
export LocalTerm, BondTerm, Operator, Hamiltonian
export XXZ, Heisenberg, Ising
export total_magnetization, staggered_magnetization, local_op, two_point
export identity_operator
export dense_matrix

# ==== MPO + expect ============================================================
export FiniteMPO, MPO
export expect, apply_mpo

# ==== Power Method ============================================================
export PowerMethodResult, power_method

# ==== TEBD + time evolution ===================================================
export TimeAxis, RealTime, ImaginaryTime
export Unitary, HermitianPSD
export Propagator, opclass, gate
export ConstantProtocol, total_time
export bond_hamiltonian
export apply_gate
export TrotterSubstep, SuzukiTrotter, trotter_steps, trotter_step

# ==== Quench + solve interface ================================================
export neel_state
export Quench, TEBD, NoTracker, Tracker
export QuenchResult
export solve

# ==== ExactDiagonalization ====================================================
export GroundState, ExactDiagonalization, EDResult, sparse
export StatevectorState, as_statevector, EDTimeResult

# ==== Disorder ================================================================
export Uniform, disorder_realization, parameter_sweep

end
