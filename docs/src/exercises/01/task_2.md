```@meta
EditURL = "task_2.jl"
```

!!! question "Task 1.2 — SVD a matrix"
    Perform an SVD on the matrix $A$ given in Moodle as `A.txt` and find the Schmidt rank needed if singular values below $10^{−3}$ are discarded.

A workaround to ensure that the data can be read during local testing as well as pages deployment build

````julia
DATA_ROOT = normpath(joinpath(@__FILE__, ".."));
FPATH_A = normpath(joinpath(DATA_ROOT, "A.txt"));
````

````julia
using DelimitedFiles

A_mat = readdlm(FPATH_A);
````

Compute the singular value decomposition (SVD) of `A_mat` using the custom function [`factorize_with_svd`](@ref)

````julia
using Qritical: factorize_with_svd
````

````julia
tolerance = 10E-3;
left_singular_mat, singular_mat, right_singular_mat = factorize_with_svd(A_mat; discard_below_threshold=true, threshold=tolerance);
````

````
[ Info: Discarding singular values below 0.01
┌ Info: Schmidt rank
│   before = 64
└   after = 6

````

# Sources
[1] L. N. Trefethen, Numerical Linear Algebra (Society for Industrial and Applied Mathematics (SIAM, 3600 Market Street, Floor 6, Philadelphia, PA 19104), Philadelphia, Pa, 1997).
[trefethen_1997](@cite)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

