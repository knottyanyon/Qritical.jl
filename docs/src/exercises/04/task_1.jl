# # Task 4.2 — MPS Overlap

# !!! question "Task 4.2 — MPS Overlap"
#     Write a function that receives two MPS of equal length and returns their overlap ``\langle \Psi_1 | \Psi_2 \rangle`` in an efficient way.

using Qritical: QriticalUtils
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI1 = normpath(joinpath(DATA_ROOT, "psi1.jls"))
FPATH_PSI2 = normpath(joinpath(DATA_ROOT, "psi2.jls"))
#--
