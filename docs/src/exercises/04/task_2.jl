# # Task 4.3 — Observables

# !!! question "Task 4.3 — Observables"
#     Write a function that receives an MPS in mixed canonical form and evaluates expectation values of ``\sigma_x`` and ``\sigma_z`` efficiently at the site where the normalization switches from left to right.

using Qritical: QriticalUtils
using LinearAlgebra
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI1 = normpath(joinpath(DATA_ROOT, "psi1.jls"))
#--

# !!! subquestion
#     **A)** Expectation value of ``\sigma_z``

# !!! subquestion
#     **B)** Expectation value of ``\sigma_x``
