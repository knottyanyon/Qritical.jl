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

include("tix.jl")
include("multix.jl")

export AbstractIx, TIx, MulTIx, dim, label, ixs, ixs_range

end
