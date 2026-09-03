#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: N/A
  coauthor:
  reviewer:
refs: N/A
credits: N/A
=#

module Operations

using TensorKit
using PrecompileTools: PrecompileTools
using Glossaries
Glossaries.@Glossary()   # own glossary namespace, separate from every other submodule's

import ..Core: PhysicalLeg
import ..Processes:
    AbstractProcess,
    QProcess,
    Scalar,
    tensor,
    outputs,
    inputs,
    is_unitary,
    is_isometry,
    dagger,
    identity_process
import ..Subroutines:
    to_mpo,
    ProductFormula,
    LieTrotter,
    SuzukiTrotter,
    sequence,
    split_commuting_groups,
    local_error_bound,
    accumulate_trotter_error!,
    TrotterErrorAccumulator,
    GaugeForm,
    LeftCanonical,
    RightCanonical,
    MixedCanonical,
    VidalGauge,
    BoundarySupport,
    TensorTrain,
    MPState,
    MPOperator,
    apply,
    overlap,
    local_expectation_value,
    _reconstruct_left_canonical,
    AutomatonTerm,
    build_automaton,
    materialize,
    close_transducer_boundary
import ..SimStudy:
    SimStudy,
    RecordingTrait,
    Active,
    Inactive,
    step!,
    finalize!,
    AbstractCollector,
    NoOpCollector

include("observable.jl")
include("hamiltonian.jl")
include("local_observable.jl")
include("propagator.jl")
include("../utils/glossary/operations.jl")   # terms must be defined before trotter_gates.jl's docstrings interpolate them
include("trotter_gates.jl")
include("expectation_values.jl")

export Observable, Hamiltonian, LocalObservable
export symmetry_group
export Time, RealTime, ImaginaryTime, Propagator
export propagator, trotterize
export TrotterGateBlock, TrotterStep, trotter_error, record_trotter_error!
export evaluate_expectation_value
export ExpectationValueSnapshot, evaluate_expectation_values

end
