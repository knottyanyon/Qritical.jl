```@meta
EditURL = "task_5.jl"
```

# Task 1.5 — Contractions

!!! question "Task 1.5"
    Generate two random matrices ``A``, ``B``, each of size ``N \times N``, and
    compute ``C_{ij} = A_{ik} B_{kj}`` (standard matrix product):
    - **(a)** once without any libraries;
    - **(b)** once using `LinearAlgebra` / BLAS.
    Compare run-time and scaling in ``N``.  Fit ``f(N) = aN^x + b``.

```julia
using LinearAlgebra, BenchmarkTools, LsqFit, CairoMakie
```

## (a) Naive triple-loop

**Your implementation:** fill in the body of `matmul_naive`.
Julia stores matrices in **column-major** order (like Fortran), so the
innermost loop should sweep a *contiguous* memory direction to avoid cache
misses.  For ``C[i,j] += A[i,k] * B[k,j]``, the contiguous dimensions are
*columns* of ``A`` (fixed ``k``) and *columns* of ``C`` (fixed ``j``).
A loop ordering of ``k → j → i`` keeps both innermost accesses contiguous.

```julia
function matmul_naive(A::AbstractMatrix, B::AbstractMatrix)
    m = size(A, 1)
    n = size(B, 2)
    p = size(A, 2)   # shared (contracted) dimension
    C = zeros(eltype(A), m, n)
    # TODO: implement C = A * B using nested for-loops (no library calls).
    # Hint: try k → j → i loop ordering for better cache use on column-major arrays.
    for k in 1:p, j in 1:n, i in 1:m
        C[i, j] += A[i, k] * B[k, j]
    end
    return C
end
```

## (b) BLAS (LinearAlgebra)

`A * B` dispatches to LAPACK/BLAS `dgemm`, which uses tiled blocking,
SIMD vectorisation, and multi-threading.  There is no kernel to write.

## Benchmark

```julia
Ns = [4, 8, 16, 32, 64, 128, 256]

times_naive = Float64[]
times_blas  = Float64[]

for N in Ns
    A = randn(N, N)
    B = randn(N, N)
    push!(times_naive, @belapsed(matmul_naive($A, $B), seconds=1))
    push!(times_blas,  @belapsed($A * $B,              seconds=1))
    println("N=$N:  naive=$(round(times_naive[end]*1e6, digits=2)) µs  BLAS=$(round(times_blas[end]*1e6, digits=2)) µs")
end
```

## Fit ``f(N) = aN^x + b``

```julia
@. model(n, p) = p[1] * n ^ p[2] + p[3]

fit_naive = curve_fit(model, Float64.(Ns), times_naive, [1e-10, 3.0, 0.0])
fit_blas  = curve_fit(model, Float64.(Ns), times_blas,  [1e-12, 3.0, 0.0])
x_naive   = round(coef(fit_naive)[2]; digits=2)
x_blas    = round(coef(fit_blas)[2];  digits=2)
```

```julia
fig = Figure(size=(700, 380))
ax  = Axis(fig[1, 1];
    title  = "Matrix multiply scaling: naive vs BLAS",
    xlabel = "N",
    ylabel = "time (s)",
    xscale = log10,
    yscale = log10,
)
scatterlines!(ax, Ns, times_naive; color=:crimson,    label="naive  ∼ O(N^$x_naive)")
scatterlines!(ax, Ns, times_blas;  color=:dodgerblue, label="BLAS   ∼ O(N^$x_blas)")
axislegend(ax; position=:lt)
fig
```

## Notes

- The naive loop is theoretically ``O(N^3)``; the fitted exponent should be
  close to 3 once ``N`` is large enough that the algorithm dominates
  over constant overheads.
- BLAS typically shows an apparent exponent ``< 3`` in benchmarks at
  moderate ``N`` because its tiled blocking means more work per cache line
  — the *effective* constant ``a`` is orders of magnitude smaller.

!!! tip "How does BLAS get ``O(N^3)`` but so much faster?"
    Matrix multiplication is nothing but a collection of dot products.
    BLAS reorganises them into *block matrix multiplications* that fit in L1/L2
    cache, then applies SIMD (AVX-512 on modern x86) to execute 8–16 FMAs
    per clock cycle.  The algorithm is still ``O(N^3)`` in FLOPs; the constant
    factor is just much smaller.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

