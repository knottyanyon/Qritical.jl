#=META
source:
  author: Claude Sonnet 5
  coauthor: N/A
  reviewer: 
docstrings:
  author: N/A
  coauthor:
  reviewer:
refs: N/A
credits: N/A
=#

module Processes

using TensorKit
using Glossaries
Glossaries.@Glossary()   # own glossary namespace, separate from Qritical's/Core's (see Core's pattern)

import ..Core: AbstractIx, TIx, LegRole, PhysicalLeg, space

include("../utils/glossary/processes.jl")   # terms must be defined before qprocess.jl's docstrings interpolate them
include("qprocess.jl")
include("categorical.jl")

export AbstractProcess, QProcess, State, Effect, Scalar
export codomain_legs, domain_legs, is_state, is_effect, equal_up_to_scalar
export value, tensor, outputs, inputs
export dagger, identity_process, is_isometry, is_unitary

end
