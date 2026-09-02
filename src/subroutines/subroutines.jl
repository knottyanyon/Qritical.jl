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

module Subroutines

using TensorKit
using Glossaries
Glossaries.@Glossary()   # own glossary namespace, separate from Qritical's/Core's/Processes'/SimStudy's (see Core's pattern)

import ..Core: AbstractIx, TIx, dim, space, LegRole, PhysicalLeg, VirtualLeg, PenroseLabel
import ..Processes:
    AbstractProcess,
    QProcess,
    State,
    Effect,
    Scalar,
    tensor,
    outputs,
    inputs,
    is_isometry,
    identity_process
import ..SimStudy:
    SimStudy,
    RecordingTrait,
    Active,
    Inactive,
    step!,
    record!,
    finalize!,
    AbstractCollector,
    NoOpCollector,
    AbstractErrorAccumulator,
    NoOpErrorAccumulator,
    QuadratureTruncationErrorAccumulator

include("../utils/glossary/subroutines.jl")   # terms must be defined before spectrum.jl's/decompositions.jl's/gauge.jl's/canonical_decompositions.jl's docstrings interpolate them
include("spectrum.jl")
include("decompositions.jl")
include("gauge.jl")
include("canonical_decompositions.jl")
include("mpoperator.jl")
include("automaton.jl")
include("../utils/glossary/trotterization.jl")   # terms must be defined before trotterization.jl's docstrings interpolate them
include("trotterization.jl")

export SingValSpectrum
export schmidt_rank, spectral_gap, entanglement_entropy, entanglement_spectrum
export local_truncation_error, global_truncation_error

export MatrixFactorization, SVDBased, QRBased, LQBased
export ExactDecomposition, SVDFACTORIZER, QRFACTORIZER, LQFACTORIZER
export factorize_tensor, factorize_tensor!
export AccessEntanglementSpectrumData, HasEntanglementSpectrum, NoEntanglementSpectrum
export SweepDirection, LeftRight, RightLeft
export reabsorb, advance_bond!, orthogonalize, orthonormalize   # `step` (Base.step, extended) isn't re-exported, same as adjoint/∘ in Processes

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

end
