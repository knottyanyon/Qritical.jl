module Qritical

include("tensor_index.jl")
include("tensor_core.jl")

export AbstractIndex, ndim
export IndexLoc, Upper, Lower
export TIx, upper, lower, uppers, lowers
export MultiIx
export IndexedTensor

end
