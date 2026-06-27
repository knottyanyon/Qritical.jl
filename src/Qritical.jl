module Qritical

using LinearAlgebra
using TensorOperations

# Abstract supertype that geometry files extend
abstract type AbstractGeometry end

include("geometry.jl")
include("dof.jl")
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
export SpinlessFermion, Electron, Majorana, HardCoreBoson
export Statistics, Commuting, Anticommuting
export local_dim, statistics, operators, physical_space
export NoSymmetry

# ==== Operator / Hamiltonian ==================================================
export uniform
export LocalTerm, BondTerm, Operator, Hamiltonian
export XXZ, Heisenberg, Ising
export total_magnetization, staggered_magnetization, local_op, two_point
export identity_operator
export dense_matrix

# ==== MPO + expect ============================================================
export FiniteMPO, MPO
export expect

end
