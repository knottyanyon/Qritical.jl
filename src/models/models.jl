#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs: N/A
credits: N/A
=#

module Models

using TensorKit
using Glossaries
Glossaries.@Glossary()   # own glossary namespace, separate from every other submodule's

import ..Core: PhysicalLeg, VirtualLeg
import ..Processes: QProcess, State
import ..Subroutines: AutomatonTerm, MPState, LeftCanonical, FiniteSupport, to_mps
import ..Operations: Hamiltonian

include("../utils/glossary/models.jl")   # terms must be defined before xxz.jl's docstrings interpolate them
include("xxz.jl")

export xxz_hamiltonian, neel_state, domain_wall_state

end
