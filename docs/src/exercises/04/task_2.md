```@meta
EditURL = "task_2.jl"
```

# Task 4.3 — Observables

!!! question "Task 4.3 — Observables"
    Write a function that receives an MPS in mixed canonical form and evaluates expectation values of ``\sigma_x`` and ``\sigma_z`` efficiently at the site where the normalization switches from left to right.

````julia
using Qritical: QriticalUtils
using LinearAlgebra
````

````julia
DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI1 = normpath(joinpath(DATA_ROOT, "psi1.jls"))
````

````
"/Users/bavithra/Documents/Uni/Courses/26_HTN/Qritical.jl/docs/src/exercises/04/psi1.jls"
````

!!! subquestion
    **A)** Expectation value of ``\sigma_z``

!!! subquestion
    **B)** Expectation value of ``\sigma_x``

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

