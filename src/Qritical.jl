module Qritical

using LinearAlgebra
using TensorOperations
using SparseArrays: SparseArrays
import SparseArrays: sparse

# include("utils/glossary/_core.jl") # commented out: reducing Qritical.jl to core-only for now
include("core/core.jl")
import .Core:
    AbstractIx,
    TIx,
    MulTIx,
    dim,
    label,
    space,
    LegRole,
    PhysicalLeg,
    VirtualLeg,
    StructureInfo,
    SymmetryStructure,
    CarriesSymmetryInfo,
    NoSymmetryInfo,
    EntanglementStructure,
    CarriesEntanglementInfo,
    NoEntanglementInfo,
    symmetry_structure,
    entanglement_structure,
    carries_symmetry_info,
    carries_entanglement_info,
    PenroseOrientation,
    Normal,
    Dual,
    PenroseLabel,
    orientation_dual,
    Leg
include("processes/processes.jl")
import .Processes:
    AbstractProcess,
    QProcess,
    State,
    Effect,
    Scalar,
    codomain_legs,
    domain_legs,
    is_state,
    is_effect,
    equal_up_to_scalar,
    value,
    tensor,
    outputs,
    inputs,
    dagger,
    identity_process,
    is_isometry,
    is_unitary
include("simstudy/simstudy.jl")
import .SimStudy:
    RecordingTrait,
    Active,
    Inactive,
    AbstractCollector,
    NoOpCollector,
    step!,
    finalize!,
    AbstractErrorAccumulator,
    NoOpErrorAccumulator,
    record!,
    QuadratureTruncationErrorAccumulator
include("subroutines/subroutines.jl")
import .Subroutines:
    SingValSpectrum,
    schmidt_rank,
    spectral_gap,
    entanglement_entropy,
    entanglement_spectrum,
    local_truncation_error,
    global_truncation_error,
    MatrixFactorization,
    SVDBased,
    QRBased,
    LQBased,
    ExactDecomposition,
    SVDFACTORIZER,
    QRFACTORIZER,
    LQFACTORIZER,
    factorize_tensor,
    factorize_tensor!,
    AccessEntanglementSpectrumData,
    HasEntanglementSpectrum,
    NoEntanglementSpectrum,
    SweepDirection,
    LeftRight,
    RightLeft,
    reabsorb,
    advance_bond!,
    orthogonalize,
    orthonormalize,
    GaugeForm,
    LeftCanonical,
    RightCanonical,
    MixedCanonical,
    VidalGauge,
    UnknownGauge,
    GaugeFreedom,
    Fixed,
    Free,
    BoundarySupport,
    FiniteSupport,
    InfiniteSupport,
    TensorTrain,
    MPState,
    MPOperator,
    is_gauge_fixed,
    CanonicalizeConfig,
    LeftCanonicalize,
    RightCanonicalize,
    SiteCanonicalize,
    to_mps,
    canonicalize,
    to_vidal,
    is_canonical,
    to_mpo,
    to_choi,
    to_operator,
    AutomatonTerm,
    HamiltonianAutomaton,
    build_automaton,
    materialize,
    ApproximateDecomposition,
    ProductFormula,
    LieTrotter,
    SuzukiTrotter,
    Suzuki4th,
    Trotterization,
    OperatorSplitting,
    sequence,
    local_error_bound,
    TrotterErrorAccumulator,
    accumulate_trotter_error!
include("operations/operations.jl")
import .Operations:
    Observable,
    Hamiltonian,
    Correlator,
    Time,
    RealTime,
    ImaginaryTime,
    Propagator,
    trotterize,
    evaluate_expectation_value,
    ExpectationValueSnapshot,
    evaluate_expectation_values
# include("utils/io.jl") # commented out: reducing Qritical.jl to core-only for now
# include("utils/deprecations.jl") # commented out: reducing Qritical.jl to core-only for now

# SECTION -  Index layer
export AbstractIx, TIx, MulTIx
export dim, label, space
export LegRole, PhysicalLeg, VirtualLeg
export StructureInfo, SymmetryStructure, CarriesSymmetryInfo, NoSymmetryInfo
export EntanglementStructure, CarriesEntanglementInfo, NoEntanglementInfo
export symmetry_structure, entanglement_structure
export carries_symmetry_info, carries_entanglement_info
export PenroseOrientation, Normal, Dual, PenroseLabel, orientation_dual, Leg

# SECTION -  Process layer
export AbstractProcess, QProcess, State, Effect, Scalar
export codomain_legs, domain_legs, is_state, is_effect, equal_up_to_scalar
export value, tensor, outputs, inputs
export dagger, identity_process, is_isometry, is_unitary

# SECTION -  SimStudy layer
export RecordingTrait, Active, Inactive
export AbstractCollector, NoOpCollector, step!, finalize!
export AbstractErrorAccumulator,
    NoOpErrorAccumulator, record!, QuadratureTruncationErrorAccumulator

# SECTION -  Subroutines layer
export SingValSpectrum
export schmidt_rank, spectral_gap, entanglement_entropy, entanglement_spectrum
export local_truncation_error, global_truncation_error
export MatrixFactorization, SVDBased, QRBased, LQBased
export ExactDecomposition, SVDFACTORIZER, QRFACTORIZER, LQFACTORIZER
export factorize_tensor, factorize_tensor!
export AccessEntanglementSpectrumData, HasEntanglementSpectrum, NoEntanglementSpectrum
export SweepDirection, LeftRight, RightLeft
export reabsorb, advance_bond!, orthogonalize, orthonormalize
export GaugeForm, LeftCanonical, RightCanonical, MixedCanonical, VidalGauge, UnknownGauge
export GaugeFreedom, Fixed, Free
export BoundarySupport, FiniteSupport, InfiniteSupport
export TensorTrain, MPState, MPOperator, is_gauge_fixed
export CanonicalizeConfig, LeftCanonicalize, RightCanonicalize, SiteCanonicalize
export to_mps, canonicalize, to_vidal, is_canonical
export to_mpo, to_choi, to_operator
export AutomatonTerm, HamiltonianAutomaton, build_automaton, materialize
export ApproximateDecomposition
export ProductFormula, LieTrotter, SuzukiTrotter, Suzuki4th
export Trotterization, OperatorSplitting
export sequence, local_error_bound, TrotterErrorAccumulator, accumulate_trotter_error!

# SECTION -  Operations layer
export Observable, Hamiltonian, Correlator
export Time, RealTime, ImaginaryTime, Propagator
export trotterize
export evaluate_expectation_value
export ExpectationValueSnapshot, evaluate_expectation_values

# Everything below is commented out: reducing Qritical.jl to core-only for now, pending the DoF
# API migration. These export non-core symbols whose defining includes are also commented out
# above.

# # SECTION -  QTensor + partitions
# export bond_label
# export QTensor, QTensor_state, QTensor_operator, dagger, to_array
# export Partition, Bipartition, complement, bipartition, group_legs
#
# # SECTION -  SVD + truncation
# export AbstractTrunc, NoTrunc, MaxBondDimTrunc, ValCutoffTrunc
# export FullSVD, ReducedSVD, do_svd
#
# # SECTION -  Spectrum + orthogonality centre
# export Bond, OrthoCenter, BondCenter, SiteCenter
# export AbstractSpectrum, SingValSpectrum, EigValSpectrum, SchmidtSpectrum
# export schmidt_rank, spectral_gap, schmidt_values
# export entanglement_entropy, entanglement_spectrum
#
# # SECTION -  State utilities + I/O
# export bipartition_matrix, as_state, load_array
#
# # SECTION -  Geometry
# export AbstractLayout, Chain, sites, bonds
#
# # SECTION -  DoF layer
# export AbstractDoF
# export Spin, SpinHalf, SpinOne
# export SpinlessFermion, Electron, MajoranaFermion, HardCoreBoson
# export CanonicalRelation, CCR, CAR
# export local_dim, canonical_relation, algebra_generators
# # SECTION -  Symmetry tags
# export NoSymmetry, physical_space
#
# # SECTION -  LatticeOperator / Hamiltonian
# export uniform_coupling
# export OneSiteTerm, TwoSiteTerm, LatticeOperator, Hamiltonian
# export XXZ, Heisenberg, Ising
# export total_magnetization, staggered_magnetization, op_at_site, two_site_op
# export identity_operator
# # SECTION -  Storage format tags
# export StorageFormat, DenseFormat, SparseFormat
# export matrix_repr
#
# # SECTION -  ExactDiagonalization
# export GroundState, ExactDiagonalization, EDResult
# export StatevectorState, as_statevector, EDTimeResult
#
# # SECTION -  Disorder
# export Uniform, disorder_realization, parameter_sweep
#
# # SECTION -  Study
# export StudyType, StaticsStudy, DynamicsStudy

end
