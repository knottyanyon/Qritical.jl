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

# # Ex 8. Real-Time TEBD: Néel Quench under XXZ
#
# ```@raw html
# <details class="page-info-drawer">
#   <summary>
#     <span class="pi-toggle-label">Page info</span>
#     <span class="pi-pills">
#       <span class="status-pill status-draft">draft</span>
#       <span class="status-pill status-needs-rewrite">needs rewrite</span>
#       <span class="status-pill status-needs-proofreading">needs proofreading</span>
#     </span>
#   </summary>
#   <div class="page-info-body">
#     <table class="page-info-table"><tbody>
#       <tr><td class="pi-key">status</td><td class="pi-val"><span class="status-pill status-draft">draft</span> <span class="status-pill status-needs-rewrite">needs rewrite</span> <span class="status-pill status-needs-proofreading">needs proofreading</span></td></tr>
#       <tr><td class="pi-key">last updated</td><td class="pi-val">2026-07-20</td></tr>
#       <tr><td class="pi-key">written by</td><td class="pi-val">Bavithra Govintharajah</td></tr>
#       <tr><td class="pi-key">edited by</td><td class="pi-val">Claude Sonnet 4.6 — initial draft</td></tr>
#     </tbody></table>
#   </div>
# </details>
# ```
#
# **Week 8 — the payoff of the whole time-evolution track.** Weeks 6–7 built the
# machinery (MPO, gates, Trotter steps); this week we point it at a genuine
# non-equilibrium problem — a **quantum quench**. Prepare a simple product state
# that is *not* an eigenstate of $H$, switch $H$ on, and watch it relax. This is the
# real-time twin of the imaginary-time (finite-temperature) evolution that Week 8's
# reading (Schollwöck §7.2) covers: the same TEBD engine, run along the real axis
# instead of the imaginary one.
#
# We start the Néel product state $|\uparrow\downarrow\uparrow\downarrow\cdots\rangle$
# and quench under the XXZ Hamiltonian.  Two signatures of many-body dynamics are
# monitored:
# - **Staggered magnetization** $M_{\rm stag}(t) = \sum_i(-1)^i \langle S^z_i\rangle$
#   decays from its initial value $-L/2$ as spins begin to flip.
# - **Entanglement entropy** $S(t)$ at the central bond grows linearly with time
#   (ballistic spreading of entanglement).
#
# !!! info "Why the entropy grows linearly — the entanglement light cone"
#     The quench injects energy locally and sets off correlations that spread at a
#     finite maximum speed (the **Lieb–Robinson** velocity) — an emergent "light
#     cone" for a non-relativistic lattice. Across the central cut, the number of
#     entangled quasiparticle pairs straddling the bond grows linearly in time, so
#     the entanglement entropy climbs like $S(t)\approx v_E\,t$. That linear growth
#     is exactly what makes real-time evolution *hard*: the required bond dimension
#     is $\chi\sim 2^{S(t)}\sim e^{v_E t}$, so a fixed cap $D$ can only follow the
#     dynamics up to a finite time — the "entanglement barrier" that bounds every
#     TEBD simulation.

# For each L: build Néel state, run TEBD, record Mstag and central-bond entropy.
# Both observables are computed in one pass — no need to re-run the evolution.
# We sweep several chain lengths so the *finite-size* fingerprints (below) are
# visible: short chains equilibrate and revive sooner, and their entropy saturates
# earlier because their maximal Schmidt rank $2^{L/2}$ is smaller than the cap $D$.
Mstag_data = Dict{Int, Vector{Float64}}()
S_data     = Dict{Int, Vector{Float64}}()
times      = dt .* (0:nsteps)   # include t=0
for L in L_values
    g = Chain(L)
    H = XXZ(g; J=1.0, Jz=1.0, h=0.0)
    ψ_t    = canonicalize(neel_state(g; dof=dof), LeftCanonical())
    W_stag = MPO(staggered_magnetization(g; dof=dof))
    W_Sz   = MPO(total_magnetization(g; dof=dof))
    #= t=0: product state has Mstag = -L/2 and S = 0 =#
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

# Sanity check: at $t=0$ the product state gives $M_{\rm stag}(0) = -L/2$ exactly.

println("Summary at t=0:")
for L in L_values
    println("  L=$L  Mstag(0)=", Mstag_data[L][1], "  (expected ", -L/2, ")  S(0)=", S_data[L][1])
end

# Staggered magnetisation decay for each chain length.

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

# Entanglement entropy growth at the central bond, for the same chain lengths.

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

# **Extracting the entanglement velocity.** The slope of $S(t)$ at short times *is*
# the entanglement velocity $v_E$ — a physical rate set by the model, not the
# algorithm. Because it is a property of the bulk dynamics, it should be essentially
# **$L$-independent**: every chain grows entanglement at the same rate until finite
# size or the bond cap intervenes. We fit a straight line over the first ten steps
# ($t\in[0.1,1.0]$), skipping $t=0$ where $S=0$ holds trivially.
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

# !!! note "Reading the finite-size fingerprints"
#     Two effects are worth hunting for in the plots:
#       1. **Staggered magnetisation** decays *faster* for smaller $L$: a small
#          system equilibrates sooner, and once a correlation front has traversed
#          the chain and reflected off the boundaries it can **revive**, visible for
#          the shortest chains within $t\lesssim 3$.
#       2. **Entanglement entropy** grows at the *same* short-time slope $v_E$ for
#          every $L$ (the velocity is a bulk property), but **saturates earlier** for
#          shorter chains, because the accessible entropy is capped at
#          $\log_2\min(2^{L/2}, D)$. This is why $D=64$ is irrelevant for $L=4$
#          (maximal Schmidt rank $2^{2}=4$) yet actively limits $L=12$.
#
# !!! info "Implementation note: one pass, two observables"
#     $M_{\rm stag}(t)$ and $S(t)$ are read off the *same* evolved state each step —
#     the staggered magnetisation from an MPO expectation value, the entropy straight
#     from the central bond's Schmidt spectrum after re-centring with
#     `BondCanonical(L÷2)`. Running the Trotter loop once and harvesting both avoids
#     evolving the state twice, and starting `times` at $t=0$ anchors the plots at
#     the exact product-state values $M_{\rm stag}(0)=-L/2$, $S(0)=0$.

# ### Try it yourself — interactive Néel quench explorer
#md # ```@raw html
#md # <iframe
#md #   src="../../../../assets/interactives/tebd_explorer.html"
#md #   style="width:100%;height:440px;border:none;border-radius:10px;margin:1.5rem 0;display:block;"
#md #   title="TEBD Néel quench — interactive D and dt explorer"
#md # ></iframe>
#md # ```

#
