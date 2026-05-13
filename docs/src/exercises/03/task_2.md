```@meta
EditURL = "task_2.jl"
```

# Task 3.2 — From Right to Left

!!! question "Task 3.2 — From Right to Left"
    Write a function that takes a right canonical representation of `psi.jls` and transforms it into a left canonical one *without* recovering the full wave-function as an intermediate step. The function should allow for a maximum matrix dimension ``D`` to truncate the state after each SVD.

````julia
using Qritical: QriticalUtils, factorize_with_svd
````

````julia
DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI = normpath(joinpath(DATA_ROOT, "psi.jls"))

tolerance = 1e-6
````

````
1.0e-6
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

