# # Task 3.2 — From Right to Left

# !!! question "Task 3.2 — From Right to Left"
#     Write a function that takes a right canonical representation of `psi.jls` and transforms it into a left canonical one **without** recovering the full wave-function as an intermediate step. The function should allow for a maximum matrix dimension ``D`` to truncate the state after each SVD.

using Qritical: QriticalUtils, factorize_with_svd
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI = normpath(joinpath(DATA_ROOT, "psi.jls"))

tolerance = 1e-6
#--
