#=META
source:
  author: Bavithra
  coauthor: Claude Opus 5
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
using Glossaries
Glossaries.@Glossary()   # own glossary namespace, separate from every other submodule's

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
    sequence,
    GaugeForm,
    LeftCanonical,
    RightCanonical,
    MixedCanonical,
    VidalGauge,
    BoundarySupport,
    TensorTrain,
    MPState,
    AutomatonTerm,
    build_automaton,
    materialize
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
include("correlator.jl")
include("propagator.jl")
include("expectation_values.jl")

export Observable, Hamiltonian, Correlator
export symmetry_group
export Time, RealTime, ImaginaryTime, Propagator
export propagator, trotterize
export evaluate_expectation_value
export ExpectationValueSnapshot, evaluate_expectation_values

end
