```@meta
EditURL = "1_SVD_a_matrix.jl"
```

# 1. SVD a matrix

Perform an SVD on the matrix $A$ given in Moodle as `A.txt` and find the Schmidt rank
needed if singular values below $10^{−3}$ are discarded.

````julia
DATA_ROOT = normpath(joinpath(@__FILE__, "..")) ## A workaround to ensure that the data can be read during local testing as well as pages deployment build
````

````
"/Users/bavithra/Documents/Uni/Courses/26_HTN/Qritical.jl/docs/src/exercises/01/"
````

````julia
FPATH_A = normpath(joinpath(DATA_ROOT, "A.txt"))
println("File path for A: $FPATH_A")

using LinearAlgebra
using DelimitedFiles
A_mat = readdlm(FPATH_A)

tolerance = 10E-3
````

````
0.01
````

Compute the singular value decomposition (SVD) of `A_mat` and return an SVD object.

````julia
A_svd = LinearAlgebra.svd(A_mat);
````

Store the Factorization Object

````julia
typeof(A_svd)
Σ_cleaned = A_svd.S .> tolerance
rank_Schmidt = count(Σ_cleaned)
````

````
6
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

