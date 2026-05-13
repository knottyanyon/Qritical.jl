```@meta
EditURL = "task_1.jl"
```

# Task 4.2 — MPS Overlap

!!! question "Task 4.2 — MPS Overlap"
    Write a function that receives two MPS (of equal length) and returns their overlap of the two in an efficient way.

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

