```@meta
EditURL = "task_1.jl"
```

# Task 2.1 — Left Canonical State

!!! question "Task 2.1 — Left Canonical State"
    Write a function that performs a left canonical decomposition of the state `psi.jls` of [Exercise 1](@ref "Task 1.3 — SVD a state"). The function should allow for a maximum matrix dimension ``D`` to truncate the state after each SVD.

````julia
using Qritical: QriticalUtils, factorize_with_svd, reshape_tensor_for_bipartition
````

````julia
DATA_ROOT = normpath(joinpath(@__FILE__, "..")) ## A workaround to ensure that the data can be read during local testing as well as pages deployment build


FPATH_PSI = normpath(joinpath(DATA_ROOT, "psi.jls"))

tolerance = 1e-6
````

````
1.0e-6
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

