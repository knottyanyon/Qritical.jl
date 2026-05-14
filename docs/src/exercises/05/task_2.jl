# # Task 5.2 — Observables I

# !!! question "Task 5.2 — Observables I"
#     Write a function that receives a MPS in ``\Gamma - \Lambda`` notation and evaluates expectation values
#     of ``\sigma_x`` and ``\sigma_z`` efficiently at all sites.

using Qritical: QriticalUtils
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
#--
