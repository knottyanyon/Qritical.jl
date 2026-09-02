#=META
source:
  author: Bavithra
  coauthor:
  reviewer:
docstrings:
  author: Bavithra
  coauthor:
  reviewer:
refs:
credits:
=#

module Core

using Glossaries
Glossaries.@Glossary()   # own glossary namespace, separate from Qritical's (see Models' pattern)

include("../utils/glossary/core.jl")   # terms must be defined before tix.jl's docstrings interpolate them
include("structure_traits.jl")
include("tix.jl")
include("leg.jl")

export AbstractIx, TIx, MulTIx, dim, label, space
export LegRole, PhysicalLeg, VirtualLeg
export StructureInfo, SymmetryStructure, CarriesSymmetryInfo, NoSymmetryInfo
export EntanglementStructure, CarriesEntanglementInfo, NoEntanglementInfo
export symmetry_structure, entanglement_structure
export carries_symmetry_info, carries_entanglement_info
export PenroseOrientation, Normal, Dual, PenroseLabel, orientation_dual, Leg

end
