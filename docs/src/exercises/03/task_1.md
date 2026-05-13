```@meta
EditURL = "task_1.jl"
```

# Task 3.1 — From Left to Right

!!! question "Task 3.1 — From Left to Right"
    Write a function that takes a left canonical representation of `psi.jls` and transforms it into a right canonical one *without* recovering the full wave-function as an intermediate step. The function should allow for a maximum matrix dimension ``D`` to truncate the state after each SVD.

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

