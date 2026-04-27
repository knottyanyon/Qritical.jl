# #1.4. Contractions

# Generate two random matrices $A, B$ each of size $N \times N$ and calculate the product
# $C_{i,j} = A_{i,k} B_{k,j}$,
# - (a) once without using any libraries
# - (b) once using a library of your choice
##
using DotEnv
ENVCFG = DotEnv.config(joinpath(ENV["PROJECT_ROOT"], ".env")); # Loads variables from .env 

EXDIR = joinpath(ENVCFG["EXERCISES_ROOT"], "01_SVD")
##
using BenchmarkTools
using Qritical
##
Ns = [10, 20, 50, 100, 200, 500, 1000]
results = []

##
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

##
# Curve Fitting
using LsqFit

# Define the model: p[1]=a, p[2]=x, p[3]=b
@. model(n, p) = p[1] * n^p[2] + p[3]

# Initial guess: a small value for a, 3 for exponent x (since it's ijk loop), 0 for b
p0 = [1e-9, 3.0, 0.0]
fit = curve_fit(model, Ns, results, p0)
a, x, b = coef(fit)

##
using CairoMakie
##

fig = Figure()
ax_1 = Axis(
    fig[1, 1];
    title="Complexity Fit: f(N) = aNˣ + b",
    xlabel="Matrix Size (N)",
    ylabel="Time (seconds)",
)

# Scatter the measured points
scatter!(ax_1, Ns, results; color=:blue, markersize=15, label="Measured")

#  Smooth line for the fitted curve
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

# Comparing with theoretical reference log axes visualization
# 2. Setup the Axis with Log10 scaling
ax_2 = Axis(
    fig[1, 2];
    title="Performance Scaling: O(N³) Complexity",
    xlabel="Matrix Size (N)",
    ylabel="Time (seconds)",
    xscale=log10,
    yscale=log10,
    xgridvisible=true,
    ygridvisible=true,
    xticks=Ns, # Show the specific N values on the axis
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
# Anchor the reference line to the first data point for comparison
ref_O3 = [results[1] * (n / Ns[1])^3 for n in Ns]

line_ref = lines!(
    ax, Ns, ref_O3; color=:red, linestyle=:dash, linewidth=2, label="Theoretical O(N³)"
)

axislegend(ax; position=:lt) # :lt = Left Top

display(fig)