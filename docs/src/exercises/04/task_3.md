```@meta
EditURL = "task_3.jl"
```

# Task 4.4 — Adding MPS

!!! question "Task 4.4 — Adding MPS"
    Write a function that receives two MPSs ``(|\Psi_1\rangle, |\Psi_2\rangle)`` and two weights ``(a, b)`` and returns their weighted sum ``a|\Psi_1\rangle + b|\Psi_2\rangle`` as an MPS.

````julia
using Qritical: QriticalUtils
````

````julia
DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI1 = normpath(joinpath(DATA_ROOT, "psi1.jls"))
FPATH_PSI2 = normpath(joinpath(DATA_ROOT, "psi2.jls"))
````

````
"/Users/bavithra/Documents/Uni/Courses/26_HTN/Qritical.jl/docs/src/exercises/04/psi2.jls"
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

