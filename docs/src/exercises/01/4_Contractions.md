```@meta
EditURL = "4_Contractions.jl"
```

#1.4. Contractions

Generate two random matrices $A, B$ each of size $N \times N$ and calculate the product
$C_{i,j} = A_{i,k} B_{k,j}$,
- (a) once without using any libraries
- (b) once using a library of your choice

Loads variables from .env

````julia
using DotEnv
````

ENVCFG = DotEnv.config(joinpath(ENV["PROJECT_ROOT"], ".env"));

EXDIR = joinpath(ENVCFG["EXERCISES_ROOT"], "01")

````julia
using BenchmarkTools
using Qritical
using CairoMakie

Ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
results = []

for N in Ns
    # We use 'setup' to prepare data.
    bench = @benchmarkable contract_N_ijk($N, A=in_data[1], B=in_data[2], C=in_data[3]) setup = (
        in_data = setup_size_N_rand_input($N)
    )
    # Run the benchmark
    # run() returns a Trial object; we take the median time in seconds
    t = median(run(bench)).time / 1e9
    push!(results, t)
    println("N = $N: Completed in $t s")
end
````

````
N = 2: Completed in 4.1e-8 s
N = 4: Completed in 1.25e-7 s
N = 8: Completed in 1.541e-6 s
N = 16: Completed in 1.3125e-5 s
N = 32: Completed in 0.00010825 s
N = 64: Completed in 0.000883791 s
N = 128: Completed in 0.007248667 s
N = 256: Completed in 0.058443771 s
N = 512: Completed in 0.470526209 s
N = 1024: Completed in 4.1820384165 s

````

Curve Fitting

````julia
using LsqFit
````

Define the model: p[1]=a, p[2]=x, p[3]=b

````julia
@. model(n, p) = p[1] * n^p[2] + p[3]
````

````
model (generic function with 1 method)
````

Initial guess: a small value for a, 3 for exponent x (since it's ijk loop), 0 for b

````julia
p0 = [1e-9, 3.0, 0.0]
fit = curve_fit(model, Ns, results, p0)
a, x, b = coef(fit)

fig = Figure();
ax_1 = Axis(
    fig[1, 1];
    title="Complexity Fit: f(N) = aNˣ + b",
    xlabel="Matrix Size (N)",
    ylabel="Time (seconds)",
)
````

````
Makie.Axis with 0 plots:

````

Scatter the measured points

````julia
scatter!(ax_1, Ns, results; color=:blue, markersize=15, label="Measured")
````

````
Makie.Scatter{Tuple{Vector{GeometryBasics.Point{2, Float64}}}}
````

 Smooth line for the fitted curve

````julia
Ns_smooth = range(minimum(Ns), maximum(Ns); length=100)
lines!(
    ax_1,
    Ns_smooth,
    model(Ns_smooth, [a, x, b]);
    color=:red,
    linewidth=3,
    label="Fit: $(round(a, sigdigits=2))N^{$(round(x, digits=2))} + $(round(b, sigdigits=2))",
)
axislegend(ax_1; position=:lt)
````

````
Makie.Legend()
````

Comparing with theoretical reference log axes visualization
Setup the Axis with Log10 scaling

````julia
ax_2 = Axis(
    fig[1, 2];
    title="Performance Scaling: O(N³) Complexity",
    xlabel="Matrix Size (N)",
    ylabel="Time (seconds)",
    xscale=log10,
    yscale=log10,
    xgridvisible=true,
    ygridvisible=true,
    xticks=Ns,
)

line_measured = scatterlines!(
    ax_2,
    Ns,
    results;
    color=:blue,
    linewidth=3,
    markersize=12,
    label="Measured contract_N_ijk",
)
````

````
Makie.ScatterLines{Tuple{Vector{GeometryBasics.Point{2, Float64}}}}
````

Anchor the reference line to the first data point for comparison

````julia
ref_O3 = [results[1] * (n / Ns[1])^3 for n in Ns]

line_ref = lines!(
    ax_2, Ns, ref_O3; color=:red, linestyle=:dash, linewidth=2, label="Theoretical O(N³)"
)
axislegend(ax_2; position=:lt)
````

````
Makie.Legend()
````

fig

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

