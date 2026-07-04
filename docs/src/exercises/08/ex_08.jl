using LinearAlgebra, Qritical
using CairoMakie, LaTeXStrings

L_values = [4, 6, 8, 10, 12]
dof      = SpinHalf()
dt       = 0.1
nsteps   = 60
D        = 64
palette = [:royalblue, :teal, :olivedrab, :darkorange, :crimson]
colors  = Dict(zip(L_values, palette))
println("Chain lengths: ", L_values, "  dt=$dt  nsteps=$nsteps  D=$D  T=", dt*nsteps)

#md # Ex 8. Real-Time TEBD: Néel Quench under XXZ
#md
#md We start the Néel product state $|\uparrow\downarrow\uparrow\downarrow\cdots\rangle$
#md and quench under the XXZ Hamiltonian.  Two signatures of many-body dynamics are
#md monitored:
#md - **Staggered magnetization** $M_{\rm stag}(t) = \sum_i(-1)^i \langle S^z_i\rangle$
#md   decays from its initial value $-L/2$ as spins begin to flip.
#md - **Entanglement entropy** $S(t)$ at the central bond grows linearly with time
#md   (ballistic spreading of entanglement).

# For each L: build Néel state, run TEBD, record Mstag and central-bond entropy.
# Both observables are computed in one pass — no need to re-run the evolution.
Mstag_data = Dict{Int, Vector{Float64}}()
S_data     = Dict{Int, Vector{Float64}}()
times      = dt .* (0:nsteps)   # include t=0
for L in L_values
    g = Chain(L)
    H = XXZ(g; J=1.0, Jz=1.0, h=0.0)
    ψ_t    = canonicalize(neel_state(g; dof=dof), LeftCanonical())
    W_stag = MPO(staggered_magnetization(g; dof=dof))
    W_Sz   = MPO(total_magnetization(g; dof=dof))
    # t=0: product state has Mstag = -L/2 and S = 0
    Mstag_L = [real(expect(ψ_t, W_stag))]
    S_L     = [0.0]
    for step in 1:nsteps
        ψ_t = trotter_step(ψ_t, H, dt, SuzukiTrotter(2); trunc=MaxBondDimTrunc(D))
        ψ_c = canonicalize(ψ_t, BondCanonical(L÷2))
        push!(Mstag_L, real(expect(ψ_c, W_stag)))
        sv = ψ_c.bond_svs[L÷2 + 1].values
        p  = sv.^2 ./ sum(sv.^2)
        push!(S_L, -sum(pᵢ -> pᵢ > 0 ? pᵢ * log2(pᵢ) : 0.0, p))
        ψ_t = ψ_c
    end
    Mstag_data[L] = Mstag_L
    S_data[L]     = S_L
    println("L=$L done  Mstag(T)=", round(Mstag_L[end]; sigdigits=3),
            "  S(T)=", round(S_L[end]; sigdigits=3), " bits")
end

println("Summary at t=0:")
for L in L_values
    println("  L=$L  Mstag(0)=", Mstag_data[L][1], "  (expected ", -L/2, ")  S(0)=", S_data[L][1])
end

fig = Figure(size=(750, 400))
ax  = Axis(fig[1,1];
    title  = L"Néel quench — staggered magnetisation ($D = %$D$)",
    xlabel = L"time $t$",
    ylabel = L"M_{\mathrm{stag}}(t)",
)
hlines!(ax, [0.0]; color=:gray, linestyle=:dot)
for L in L_values
    lines!(  ax, times, Mstag_data[L]; color=colors[L], linewidth=2.0, label=L"L = %$L")
    scatter!(ax, times, Mstag_data[L]; color=colors[L], markersize=5)
end
axislegend(ax; position=:rb)
fig

fig2 = Figure(size=(750, 400))
ax2  = Axis(fig2[1,1];
    title  = L"Entanglement entropy at central bond ($D = %$D$)",
    xlabel = L"time $t$",
    ylabel = L"S(t) \; [\mathrm{bits}]",
)
for L in L_values
    lines!(  ax2, times, S_data[L]; color=colors[L], linewidth=2.0, label=L"L = %$L")
    scatter!(ax2, times, S_data[L]; color=colors[L], markersize=5)
end
axislegend(ax2; position=:lt)
fig2

# Short-time entanglement velocity: S(t) ≈ vₑ·t (ballistic spreading).
# Fit over the first 10 steps (t=0.1..1.0), excluding t=0 where S=0 trivially.
println("Short-time entanglement velocities (linear fit, t ∈ [0.1, 1.0]):")
for L in L_values
    t_fit  = times[2:11]
    S_fit  = S_data[L][2:11]
    coeffs = [ones(10) t_fit] \ S_fit
    v_E    = coeffs[2]
    println("  L=$L:  vₑ ≈ ", round(v_E; sigdigits=3), " bits/time")
end

#md Key design choices:
#md - The old notebook ran TEBD twice per L (once via solve(…, tracker=…), once in the entropy loop) — the new code does a single manual Trotter loop that computes both observables in one pass.
#md - times now starts at t=0 (product state as the zeroth point), so the initial Mstag = −L/2 and S=0 anchor is visible in the plots.
#md - Each chain length gets a distinct color from the palette dict so the legend is unambiguous.
#md
#md ★ Insight ─────────────────────────────────────
#md Finite-size effects to look for in the plots: (1) Mstag decays faster for smaller L because the finite system equilibrates sooner — for very small L the revival is visible within t=3. (2) Entropy grows linearly at short times at the same rate across all L (same entanglement velocity vₑ), but saturates earlier for smaller L once the bond dimension is capped by min(2^(L/2), D) — this is how D=64 becomes irrelevant for L=4 (max Schmidt rank = 4) while mattering for L=12.

#md
