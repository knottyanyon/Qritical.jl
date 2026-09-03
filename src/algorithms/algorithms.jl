#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs: schollwoeck_2011
credits: N/A
=#

module Algorithms

using TensorKit
using Glossaries
Glossaries.@Glossary()

import ..Core: PhysicalLeg
import ..Processes: QProcess, tensor, outputs, inputs, value
import ..Subroutines:
    GaugeForm,
    MixedCanonical,
    LeftCanonical,
    RightCanonical,
    VidalGauge,
    ProductFormula,
    TensorTrain,
    MPState,
    CanonicalizeConfig,
    MixedCanonicalize,
    canonicalize,
    is_canonical,
    apply_gate,
    norm,
    SingValSpectrum,
    entanglement_entropy,
    factorize_tensor,
    HasEntanglementSpectrum,
    AbstractErrorAccumulator,
    NoOpErrorAccumulator,
    QuadratureTruncationErrorAccumulator,
    TrotterErrorAccumulator,
    local_error_bound
import ..Operations:
    Hamiltonian,
    Time,
    RealTime,
    Propagator,
    propagator,
    trotterize,
    TrotterGateBlock,
    TrotterStep,
    trotter_error,
    evaluate_expectation_values
import ..SimStudy:
    SimStudy,
    RecordingTrait,
    Active,
    Inactive,
    step!,
    finalize!,
    AbstractCollector,
    NoOpCollector

include("../utils/glossary/algorithms.jl")
include("tebd.jl")

export TEBDAlgorithm, evolve!, TEBDStepSnapshot

end
