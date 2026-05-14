# # Task 5.4 — Observables III

# !!! question "Task 5.4 — Observables III"
#     Write a function that receives a MPS in ``\Gamma - \Lambda`` notation and evaluates the correlation
#     function ``\left\langle \sigma_z^{L/2} \sigma_z^{L/2+i} \right\rangle`` for all ``i``.

using Qritical: QriticalUtils
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
#--
