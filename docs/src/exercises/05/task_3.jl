# # Task 5.3 — Observables II

# !!! question "Task 5.3 — Observables II"
#     Write a function that receives a MPS in ``\Gamma - \Lambda`` notation and evaluates the correlation
#     function ``\langle \sigma_z^i \sigma_z^{i+1} \rangle`` efficiently at all sites.

using Qritical: QriticalUtils
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
#--
