# # 1. Left Canonical State

# Write a function that performs a left canonical decomposition of the state `psi.jls` of Ex1. The function should allow for a maximum matrix dimension $D$ to truncate the state after each SVD.

using Qritical: QriticalUtils, factorize_with_svd, reshape_tensor_for_bipartition
#--

DATA_ROOT = normpath(joinpath(@__FILE__, "..")) ## A workaround to ensure that the data can be read during local testing as well as pages deployment build


FPATH_PSI = normpath(joinpath(DATA_ROOT, "psi.jls"))

tolerance = 1e-6
#--

