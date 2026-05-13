# # Task 4.4 — Adding MPS

# !!! question "Task 4.4 — Adding MPS"
#     Write a function that receives two MPSs ``(|\Psi_1\rangle, |\Psi_2\rangle)`` and two weights ``(a, b)`` and returns their weighted sum ``a|\Psi_1\rangle + b|\Psi_2\rangle`` as an MPS.

using Qritical: QriticalUtils
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI1 = normpath(joinpath(DATA_ROOT, "psi1.jls"))
FPATH_PSI2 = normpath(joinpath(DATA_ROOT, "psi2.jls"))
#--
