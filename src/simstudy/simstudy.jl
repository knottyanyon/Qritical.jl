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

module SimStudy

using Glossaries
Glossaries.@Glossary()   # own glossary namespace, separate from Qritical's/Core's/Processes' (see Core's pattern)

include("../utils/glossary/simstudy.jl")   # terms must be defined before collectors.jl's/accumulators.jl's docstrings interpolate them
include("collectors.jl")
include("accumulators.jl")

export RecordingTrait, Active, Inactive
export AbstractCollector, NoOpCollector, step!, finalize!
export AbstractErrorAccumulator,
    NoOpErrorAccumulator, record!, QuadratureTruncationErrorAccumulator

end
