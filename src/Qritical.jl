module Qritical

using LinearAlgebra
using TensorOperations
using SparseArrays: SparseArrays
import SparseArrays: sparse

include("experimental/topograph/layout.jl")
include("experimental/min_model_kit/dof.jl")
include("experimental/min_model_kit/lattice/symmetries.jl")
include("tensors/storage_format.jl")
include("operators/operator.jl")
include("core/core.jl")
import .Core: AbstractIx, TIx, dim, label, ixs, ixs_range
include("tensors/multix.jl")
include("tensors/partition.jl")
include("tensors/qtensor.jl")
include("tensors/tensor_utils.jl")
include("tensors/bond.jl")
include("tensors/ortho_center.jl")
include("tensors/spectrum.jl")
include("experimental/topograph/gengraph.jl")
include("experimental/topograph/ordinary_graph.jl")
include("experimental/topograph/ids.jl")
include("experimental/topograph/node.jl")
# wire.jl, leg.jl, attachment.jl, orientation.jl, network.jl, pin.jl, compactify.jl,
# convention.jl still use the old variance-tagged TIx{Upper}/TIx{Lower}/IxLoc index API
# (M12, a separate milestone). Temporarily disabled while the TensorKit TensorMap migration
# is in progress so the rest of the package can load and be verified; restore once M12 is
# migrated to the flat TIx design in its own session.
# include("experimental/topograph/wire.jl")
# include("experimental/topograph/leg.jl")
# include("experimental/topograph/attachment.jl")
# include("experimental/topograph/orientation.jl")
# include("experimental/topograph/network.jl")
# include("experimental/topograph/pin.jl")
# include("experimental/topograph/compactify.jl")
# include("experimental/topograph/progressive.jl")
# include("experimental/min_model_kit/lattice/lattice_graph.jl")
# include("experimental/topograph/convention.jl")
include("tensors/svd.jl")
include("utils/io.jl")
include("states/mps.jl")
include("states/canonicalize.jl")
include("states/vidal.jl")
include("operators/correlators.jl")
include("operators/finite_mpo.jl")
include("algorithms/power_method.jl")
include("algorithms/tebd.jl")
include("studies/study.jl")
include("studies/evolution.jl")
include("algorithms/ed.jl")
include("studies/disorder.jl")
include("utils/deprecations.jl")

# SECTION -  Index layer
export AbstractIx, TIx, MulTIx
export dim, label
export ixs, ixs_range, bond_label

# SECTION -  QTensor + partitions
export QTensor, QTensor_state, QTensor_operator, dagger, to_array
export Partition, Bipartition, complement, bipartition, group_legs

# SECTION -  SVD + truncation 
export AbstractTrunc, NoTrunc, MaxBondDimTrunc, ValCutoffTrunc
export FullSVD, ReducedSVD, do_svd

# SECTION -  Spectrum + orthogonality centre 
export Bond, OrthoCenter, BondCenter, SiteCenter
export AbstractSpectrum, SingValSpectrum, EigValSpectrum, SchmidtSpectrum
export schmidt_rank, spectral_gap, schmidt_values
export entanglement_entropy, entanglement_spectrum

# SECTION -  Topograph — generalized topological graph layer 
export AbstractGenTopoGraph
export GraphTrait, Ungraded, Oriented, Polarised
export graph_trait, is_oriented, is_polarised
export WireId, LegId, NodeId
export Wire, Leg
export Attachment, Pinned, HalfLoose, Loose, Circle
export attachment
export make_leg
export LegOrientation, Incoming, Outgoing, orientation, leg_orientation
export TensorNetwork, add_node!, add_wire!, add_leg!
export nodes, wires, legs, ends, incident, degree
export attach!, pin!, cut!
export compactify, boundary, is_ordinary, is_closed
export is_progressive, has_circuit
export OrdinaryGraphNetwork, UndirectedGraph
export LinkId, LatticeGraph
export links
export qdir_string

# SECTION -  State utilities + I/O 
export bipartition_matrix, as_state, load_array

# SECTION -  MPS & canonical forms 
export AbstractMPSForm, CanonicalForm, VidalForm, ArbitraryForm
export FiniteMPS, to_mps, add_mps
export CanonicalizeConfig, LeftCanonical, RightCanonical, BondCanonical, SiteCanonical
export canonicalize, canonical_error, is_canonical, overlap, local_expectation, two_site_op
export to_vidal, to_canonical

# SECTION -  Geometry 
export AbstractLayout, Chain, sites, bonds

# SECTION -  DoF layer 
export AbstractDoF
export Spin, SpinHalf, SpinOne
export SpinlessFermion, Electron, MajoranaFermion, HardCoreBoson
export CanonicalRelation, CCR, CAR
export local_dim, canonical_relation, algebra_generators
# SECTION -  Symmetry tags 
export NoSymmetry, physical_space

# SECTION -  LatticeOperator / Hamiltonian 
export uniform_coupling
export OneSiteTerm, TwoSiteTerm, LatticeOperator, Hamiltonian
export XXZ, Heisenberg, Ising
export total_magnetization, staggered_magnetization, op_at_site, two_site_op
export identity_operator
# SECTION -  Storage format tags 
export StorageFormat, DenseFormat, SparseFormat
export matrix_repr

# SECTION -  MPO + expect 
export FiniteMPO, MPO
export expect, apply_mpo

# SECTION -  Power Method 
export PowerMethodResult, power_method

# SECTION -  TEBD + time evolution 
export TimeAxis, RealTime, ImaginaryTime
export Unitary, HermitianPSD
export Propagator, opclass, gate
export ConstantProtocol, total_time
export bond_hamiltonian
export apply_gate
export TrotterSubstep, SuzukiTrotter, trotter_steps, trotter_step

# SECTION -  Evolution + solve interface 
export neel_state
export Evolution, TEBD, NoTracker, Tracker
export EvolutionResult
export solve

# SECTION -  ExactDiagonalization 
export GroundState, ExactDiagonalization, EDResult
export StatevectorState, as_statevector, EDTimeResult

# SECTION -  Disorder 
export Uniform, disorder_realization, parameter_sweep

# SECTION -  Study 

export StudyType, StaticsStudy, DynamicsStudy

end
