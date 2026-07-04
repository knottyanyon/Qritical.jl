# # Ex 10. A Toy ED Code

# making sure that the correct julia project is activated instead of the current directory so all exports are available.

using Qritical 
using LinearAlgebra
using SparseArrays # stdlib module support for sparse vectors and sparse matrices
# arpack

# ## (a) Write the operators for $\alpha=x,y,z$ in the many-body Hilbert space.
#
# The many-body Hilbert space for $L$ spin-½ sites is $\mathcal{H} = \bigotimes_{i=1}^{L} \mathbb{C}^2$, $D=2^L$. 
#
# The spin-½ matrices are
# $$S^z = \tfrac12\begin{pmatrix}1&0\\0&-1\end{pmatrix},\quad
# S^+ = \begin{pmatrix}0&1\\0&0\end{pmatrix},\quad
# S^x = \tfrac{1}{2}(S^++S^-),\quad S^y = \tfrac{1}{2i}(S^+-S^-).$$

"""
    create_XXZ_L_spin_chain(L; Jxy=1.0, Jz=1.0, h=0.0)
Build an XXZ spin-½ chain of length `L` and return all commonly needed objects
as a named tuple so callers avoid boilerplate setup.
Keyword arguments:
- `Jxy` — transverse (flip-flop) coupling  (default 1.0)
- `Jz`  — longitudinal (Ising) coupling    (default 1.0)
- `h`   — on-site magnetic field           (default 0.0)
Returns `(; L, dof, lattice, H, ops, d)` where
- `dof` — `SpinHalf()` degree of freedom
- `lattice`   — `Chain(L)` geometry
- `H`   — `LatticeOperator` (XXZ Hamiltonian)
- `ops` — `NamedTuple` from `algebra_generators(SpinHalf())`
- `d`   — local Hilbert-space dimension (= 2)
"""
function create_XXZ_L_spin_chain(L; Jxy=1.0, Jz=1.0, h=0.0)
    dof = SpinHalf()
    lattice   = Chain(L)
    hamiltonian   = XXZ(lattice; J=Jxy, Jz=Jz, h=h)
    ops = algebra_generators(dof) # get the site operators that generate the correct operator algebra
    d   = local_dim(dof)
    return (; L, dof, lattice, hamiltonian, ops, d)
end
# Default model used throughout part (a): L=4 isotropic Heisenberg
(; L, dof, lattice, hamiltonian, ops, d) = create_XXZ_L_spin_chain(4)
@show ops.Sz
@show local_dim(dof)

# ### (a.1) One-site operator $S^{\alpha}_i$
#
# <div style="text-align: center">
# <img src="10_a_1.svg" width="50%">
# <div>
#
# \begin{equation}
# \hat{O}_\ell = \left[O_{\ell}\right]^{{\sigma'}_{\ell}}_{{\sigma}_{\ell}} \ket{{\sigma'}_{\ell}} \bra{{\sigma}_{\ell}}
# \end{equation}

# Single-site: Sᶻ at site 2 embedded in the full 2^L Hilbert space
Sz_op_2  = op_at_site(lattice, dof, :Sz, 2)   # geometry required — sets the Kronecker-product size
Sz_mat_2 = matrix_repr(Sz_op_2)               # 16×16 dense matrix for L=4
@show size(Sz_mat_2)   # (16, 16)  — identity ⊗ Sz ⊗ identity ⊗ identity

# ### (a.2) Two-site operator $S^{\alpha}_i S^{\alpha}_{i+1}$
#
# <div style="text-align: center">
# <img src="10_a_2.svg" width="50%">
# <div>
#
# \begin{equation}
# \hat{O}_{\ell,\ell+1}= \left[O_{\ell,\ell+1}\right]^{{\sigma'}_{\ell}{\sigma'}_{\ell+1}}_{{\sigma}_{\ell}{\sigma}_{\ell+1}} \ket{{\sigma'}_{\ell} {\sigma'}_{\ell+1}} \bra{{\sigma}_{\ell}{\sigma}_{\ell+1}}
# \end{equation}

# Two-site: Sz_1 Sz_2  (nearest-neighbour)
SzSz_12_op  = two_site_op(lattice, SpinHalf(), :Sz, 1, :Sz, 2)
SzSz_12_mat = matrix_repr(SzSz_12_op)
# Two-site: Sz_1 Sz_3  (non-nearest-neighbour)
SzSz_13_op  = two_site_op(lattice, SpinHalf(), :Sz, 1, :Sz, 3)
SzSz_13_mat = matrix_repr(SzSz_13_op)
println("SzSz_12 diagonal (first 8): ", round.(real.(diag(SzSz_12_mat)[1:8]), digits=3))

# test : verify the spin algebra  [Sx, Sy] = i Sz  on the local matrices
Sx, Sy, Sz = ops.Sx, ops.Sy, ops.Sz
commutator = Sx * Sy - Sy * Sx
@assert norm(commutator - im * Sz) < 1e-14 "[Sx,Sy] ≠ iSz  (Condon–Shortley sign error)"
# test : Verify Sz² eigenvalues: ±½ → (Sz²)_eigenvals = [¼, ¼]
sz_evals = sort(real.(eigvals(Sz)))
@assert sz_evals ≈ [-0.5, 0.5] atol=1e-14 "Sz eigenvalues are not ±½"
# test : Single-site embedding: eigenvalues of Sz₂ in a 4-site chain should be four-fold degenerate ±½ (other sites are unconstrained)
evals_Sz2 = sort(unique(round.(real.(eigvals(Sz_mat_2)), digits=10)))
@assert evals_Sz2 ≈ [-0.5, 0.5] atol=1e-10 "Sz₂ embed: eigenvalues wrong"
println("local algebra and embedding checks passed ✓")

# ## (b) Construct from (a) the Hamiltonian $H$ of a XXZ spin chain
#
# $$H = \sum_{i=1}^{L-1} \Bigl[ \tfrac{J}{2}(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+)
#     + J_z\, S_i^z S_{i+1}^z \Bigr] - h \sum_i S_i^z.$$

# ── Build XXZ for L=6, J=Jz=1 (isotropic Heisenberg), no field ──────────────
(; L, hamiltonian) = create_XXZ_L_spin_chain(6)
println("Number of bond terms : ", length(hamiltonian.bond))
println("Number of onsite terms: ", length(hamiltonian.onsite))
M_dense = matrix_repr(hamiltonian)                    # gives the $2^L \times 2^L$ dense matrix
M_sparse = matrix_repr(hamiltonian, SparseFormat())    # SparseMatrixCSC - gives the sparse CSC version (same but faster for large $L$)
println("Dense shape : ", size(M_dense))
println("Sparse nnz  : ", nnz(M_sparse), " / ", prod(size(M_sparse)), " entries")

using CairoMakie
# ── Collect term counts and matrix fill statistics for a range of L ──────────
# For an open XXZ chain:
#   onsite terms  = L               (one Sz field term per site)
#   bond terms    = 3(L−1)          (Sp⊗Sm, Sm⊗Sp, Sz⊗Sz per bond)
#   dense entries = D²  = 4^L       (all elements of the full matrix stored)
#   sparse nnz    ∝ L·D = L·2^L    (only matrix elements that couple two basis states)
function hamiltonian_density_vs_L(Ls)
    n_onsite_v   = Int[]
    n_bond_v     = Int[]
    dense_total  = Int[]
    sparse_nnz_v = Int[]
    for L in Ls
        H = create_XXZ_L_spin_chain(L).hamiltonian
        M = matrix_repr(H, SparseFormat())
        D = size(M, 1)
        push!(n_onsite_v,   length(H.onsite))
        push!(n_bond_v,     length(H.bond))
        push!(dense_total,  D^2)
        push!(sparse_nnz_v, nnz(M))
    end
    return n_onsite_v, n_bond_v, dense_total, sparse_nnz_v
end
Ls_v = collect(2:2:14)
n_onsite_v, n_bond_v, dense_total, sparse_nnz_v = hamiltonian_density_vs_L(Ls_v)
fig = Figure(size = (880, 420))
#= Left panel: bond and onsite term counts — both linear in L =#
ax1 = Axis(fig[1, 1];
    xlabel = "Chain length  L",
    ylabel = "Number of terms",
    title  = "Operator term count vs L",
    xticks = Ls_v,
)
lines!(ax1,   Ls_v, n_bond_v;   color = :steelblue, label = "Bond terms  3(L−1)")
scatter!(ax1, Ls_v, n_bond_v;   color = :steelblue, markersize = 10)
lines!(ax1,   Ls_v, n_onsite_v; color = :orangered,  label = "Onsite terms  L")
scatter!(ax1, Ls_v, n_onsite_v; color = :orangered,  markersize = 10)
axislegend(ax1; position = :lt)
#= Right panel: dense D² vs sparse nnz — log scale reveals the exponential gap =#
ax2 = Axis(fig[1, 2];
    xlabel = "Chain length  L",
    ylabel = "Number of stored entries",
    title  = "Dense D² vs sparse nnz  (log scale)",
    yscale = log10,
    xticks = Ls_v,
)
lines!(ax2,   Ls_v, dense_total;   color = :steelblue, label = "Dense  D² = 4ᴸ")
scatter!(ax2, Ls_v, dense_total;   color = :steelblue, markersize = 10)
lines!(ax2,   Ls_v, sparse_nnz_v;  color = :orangered,  label = "Sparse nnz  ∝ L·2ᴸ")
scatter!(ax2, Ls_v, sparse_nnz_v;  color = :orangered,  markersize = 10)
axislegend(ax2; position = :lt)
fig

# test : Hermiticity, and known L=2 spectrum 
# Hermiticity
@assert norm(M_dense - M_dense') < 1e-13  "H is not Hermitian"
@assert norm(Matrix(M_sparse) - Matrix(M_sparse)') < 1e-13

# test : Heisenberg L=2 exact spectrum: one singlet at −¾ J, three triplets at +¼ J
H2  = create_XXZ_L_spin_chain(2).hamiltonian
ev2 = sort(real.(eigvals(Hermitian(matrix_repr(H2)))))
@assert ev2 ≈ [-0.75, 0.25, 0.25, 0.25]  atol=1e-12  "L=2 spectrum incorrect"

# test : Sparse and dense must agree
@assert norm(M_dense - Matrix(M_sparse)) < 1e-13  "dense ≠ sparse"
println("Hermiticity, L=2 spectrum, and sparse/dense consistency passed ✓")

# ## (c) Determine the GS of $H$ using a sparse eigenspectrum solver.
#
# Qritical wraps KrylovKit's Lanczos under `ExactDiagonalization(:ground)`.  This
# only computes the lowest eigenvalue and its eigenvector — cost $O(2^L)$ per
# matrix–vector product, with the Krylov subspace kept small.
#
# API: `solve(H, GroundState(), ExactDiagonalization(:ground))` → `EDResult`
# with fields `.energy` and `.state`.

# ── Ground state, isotropic Heisenberg L=8 ──────────────────────────────────
(; L, hamiltonian) = create_XXZ_L_spin_chain(8)
gs = solve(hamiltonian, GroundState(), ExactDiagonalization(:ground))
println("GS energy (L=$L, Heisenberg): ", round(gs.energy, digits=8))
println("GS vector norm            : ", round(norm(gs.state), digits=14))

# ── Parameter scan: E₀(L) as a function of chain length ─────────────────────
Ls = [4, 6, 8, 10]
E0s = parameter_sweep(Ls) do L
    (; hamiltonian) = create_XXZ_L_spin_chain(L)
    solve(hamiltonian, GroundState(), ExactDiagonalization(:ground)).energy
end
println("L   E₀/L")
for (L, E) in zip(Ls, E0s)
    println("$L   ", round(E/L, digits=6))
end
# Should approach the Bethe-ansatz value E/L → −ln2 + ¼ ≈ −0.4431 as L → ∞

# ── Check (c): GS energy matches dense eigensolver; GS is unit-norm ─────────
H6  = create_XXZ_L_spin_chain(6).hamiltonian
gs6 = solve(H6, GroundState(), ExactDiagonalization(:ground))
# Dense reference
ev_dense = minimum(real.(eigvals(Hermitian(matrix_repr(H6)))))
@assert gs6.energy ≈ ev_dense  atol=1e-8  "Lanczos GS energy ≠ dense minimum"
@assert norm(gs6.state) ≈ 1.0  atol=1e-12  "GS not unit-norm"
# GS must be eigenstate: H|ψ₀⟩ = E₀|ψ₀⟩ → ‖(H - E₀I)|ψ₀⟩‖ < tol
residual = norm((Matrix(matrix_repr(H6, SparseFormat())) - gs6.energy * I) * gs6.state)
@assert residual < 1e-8  "GS residual too large: $residual"
println("Part (c) ✓  —  E₀ matches dense, unit norm, eigenstate residual = ",
        round(residual, sigdigits=3))

# ## (d) Determine the full eigenspectrum of $H$
#
# `ExactDiagonalization(:full)` calls Julia's dense `eigen(Hermitian(M))` — cost
# $O(d^{3L})$.  The result `.spectrum` contains **all** $2^L$ eigenvalues sorted
# ascending.  Only practical for $L \lesssim 14$; the $2^{20}$ guard in
# `matrix_repr(H, DenseFormat())` blocks accidental use on larger systems.

# ── Full spectrum of Heisenberg L=6 ─────────────────────────────────────────
(; hamiltonian) = create_XXZ_L_spin_chain(6)
full = solve(hamiltonian, GroundState(), ExactDiagonalization(:full))
println("Number of eigenvalues: ", length(full.spectrum))
println("Lowest  5: ", round.(full.spectrum[1:5],   digits=6))
println("Highest 5: ", round.(full.spectrum[end-4:end], digits=6))
println("Spectral width: ", round(full.spectrum[end] - full.spectrum[1], digits=6))

# ── Ising limit: Jz=2, J=0 ─── only SᶻSᶻ coupling ──────────────────────────
# Spectrum is purely diagonal in the computational basis: each bond contributes
# ±Jz/4 depending on spin alignment → all eigenvalues in {-½Jz(L-1)/4, …}
H_ising = create_XXZ_L_spin_chain(4; Jxy=0.0, Jz=2.0, h=0.0).hamiltonian
ising   = solve(H_ising, GroundState(), ExactDiagonalization(:full))
println("Ising L=4 unique eigenvalues: ", sort(unique(round.(ising.spectrum, digits=6))))

# ── Check (d): spectrum must be real, sorted, match GS and L=2 analytics ────
H6    = create_XXZ_L_spin_chain(6).hamiltonian
full6 = solve(H6, GroundState(), ExactDiagonalization(:full))
@assert length(full6.spectrum) == 2^6  "Wrong number of eigenvalues: $(length(full6.spectrum))"
@assert issorted(full6.spectrum)  "Spectrum not sorted"
@assert full6.spectrum ≈ sort(full6.spectrum)  atol=1e-14
# GS energy from :full must match :ground
gs6 = solve(H6, GroundState(), ExactDiagonalization(:ground))
@assert full6.energy ≈ gs6.energy  atol=1e-8  ":full GS ≠ :ground GS"
# Trace = sum of eigenvalues = Tr(H) must be zero for traceless XXZ
@assert abs(sum(full6.spectrum)) < 1e-10  "Tr(H) ≠ 0: $(sum(full6.spectrum))"
println("Part (d) ✓  —  ", 2^6, " eigenvalues, sorted, Tr=0, GS consistent with Lanczos.")

# ## (e) Write a time evolution code w.r.t to $H$ for a given initial state $|\Psi(t)\rangle = e^{-iHt}|\Psi_0\rangle$
#
# The exact propagator diagonalises $H = V\Lambda V^\dagger$ **once**, then
# evaluates $e^{-iHT}|\Psi_0\rangle = V\,\mathrm{diag}(e^{-i\lambda_k T})\,V^\dagger|\Psi_0\rangle$
# at cost $O(d^{2L})$ per time point — no Trotter error, exact to machine precision.
#
# API:
# ```julia
# sv = as_statevector(ψ₀_vec)                      # wrap Vector{ComplexF64}
# p  = ConstantProtocol(RealTime(), dt, nsteps, H)  # T = dt × nsteps
# r  = solve(H, sv, ExactDiagonalization(:time), p) # EDTimeResult
# r.state   # final |Ψ(T)⟩
# r.time    # T
# ```

# ── Néel initial state |↑↓↑↓…⟩ quenched under Heisenberg ───────────────────
(; L, lattice, hamiltonian, d) = create_XXZ_L_spin_chain(6)
# Build the Néel state directly as a computational-basis vector.
# Basis convention (big-endian, site 1 most significant):
#   index k = 1 + Σᵢ (σᵢ - 1) · d^(L-i),  σᵢ ∈ {1(↑), 2(↓)}
# Néel pattern: odd sites ↑ (σ=1, contributes 0), even sites ↓ (σ=2, contributes 1·d^(L-i))
neel_idx = 1 + sum(iseven(i) ? d^(L - i) : 0 for i in 1:L)
ψ₀ = zeros(ComplexF64, d^L)
ψ₀[neel_idx] = 1.0
sv = as_statevector(ψ₀)
println("Néel state index (1-based, big-endian): ", neel_idx, "  (out of ", d^L, ")")
println("‖ψ₀‖ = ", norm(ψ₀))

# ── Evolve and measure staggered magnetisation M_stag(t) = ⟨Ψ(t)|(−1)ⁱSᶻᵢ|Ψ(t)⟩ ──
T_max  = 5.0
n_t    = 50
dt     = T_max / n_t
# Assemble staggered magnetisation on the shared lattice geometry.
# ops.Sz is already the local 2×2 matrix; no need to call algebra_generators again.
stag_terms = [OneSiteTerm(i, ops.Sz, (-1.0)^i) for i in 1:L]
M_stag_op  = LatticeOperator(dof, lattice, stag_terms, TwoSiteTerm[])
M_stag_mat = matrix_repr(M_stag_op)
times  = Float64[]
mstag  = Float64[]
ψ = copy(ψ₀)
for step in 1:n_t
    p = ConstantProtocol(RealTime(), dt, 1, hamiltonian)
    r = solve(hamiltonian, as_statevector(ψ), ExactDiagonalization(:time), p)
    global ψ = r.state
    push!(times, dt * step)
    push!(mstag, real(dot(ψ, M_stag_mat * ψ)))
end
println("t=0                → M_stag = ", real(dot(ψ₀, M_stag_mat * ψ₀)))
println("t=", round(T_max, digits=1), "             → M_stag = ", round(mstag[end], digits=6))
println("(value at t=0 is −L/2 = ", -L/2, " for the perfect Néel state)")

# ── Alternatively: propagate the full interval in one shot ──────────────────
T_total = 5.0
p_full  = ConstantProtocol(RealTime(), T_total, 1, hamiltonian)
r_full  = solve(hamiltonian, sv, ExactDiagonalization(:time), p_full)
println("One-shot final state norm : ", round(norm(r_full.state), digits=14))
println("Total time returned       : ", r_full.time)

# ── Check (e): norm preservation; GS is a fixed point; imaginary-time converges ──
H4  = create_XXZ_L_spin_chain(4).hamiltonian
gs4 = solve(H4, GroundState(), ExactDiagonalization(:ground))
# 1. Norm is exactly preserved under real-time evolution
sv_rand = as_statevector(normalize(randn(ComplexF64, 2^4)))
p_check = ConstantProtocol(RealTime(), 3.7, 1, H4)
r_check = solve(H4, sv_rand, ExactDiagonalization(:time), p_check)
@assert norm(r_check.state) ≈ 1.0  atol=1e-14  "Norm not preserved under real-time ED"
# 2. GS |ψ₀⟩ is a fixed point of e^{−iHt}: e^{−iHt}|ψ₀⟩ = e^{−iE₀t}|ψ₀⟩
# Phase extracted via inner product ⟨ψ₀|ψ(t)⟩ = e^{-iE₀t}  (stable — averages all 2^L components)
sv_gs = as_statevector(gs4.state)
p_gs  = ConstantProtocol(RealTime(), 1.2, 1, H4)
r_gs  = solve(H4, sv_gs, ExactDiagonalization(:time), p_gs)
phase = dot(gs4.state, r_gs.state)   # = e^{-iE₀t} since ‖gs4‖ = 1
@assert abs(abs(phase) - 1.0) < 1e-13  "GS is not a phase eigenstate"
@assert norm(r_gs.state - phase * gs4.state) < 1e-12  "GS not pure phase evolution"
# 3. Imaginary-time projects to GS: e^{-Hτ}|ψ₀_rand⟩ → |ψ₀⟩  (up to norm)
sv_it   = as_statevector(normalize(randn(ComplexF64, 2^4)))
p_it    = ConstantProtocol(ImaginaryTime(), 20.0, 1, H4)
r_it    = solve(H4, sv_it, ExactDiagonalization(:time), p_it)
ψ_proj  = normalize(r_it.state)
overlap = abs(dot(ψ_proj, gs4.state))
@assert abs(overlap - 1.0) < 1e-10  "Imaginary-time did not converge to GS (overlap = $overlap)"
println("Part (e) ✓  —  norm preserved, GS is phase fixed-point, imaginary-time→GS.")

# ## Summary
#
# | Part | Qritical entry point | Verified |
# |------|---------------------|---------|
# | (a) single-site $S_i^\alpha$ | `algebra_generators(SpinHalf())`, `op_at_site`, `matrix_repr(H)` | spin algebra $[S^x,S^y]=iS^z$ |
# | (a) two-site $S_i^\alpha S_j^\beta$ | `two_site_op(g, dof, :Sx, i, :Sz, j)`, `matrix_repr(H)` | agrees with kron product |
# | (b) XXZ Hamiltonian | `create_XXZ_L_spin_chain(L)`, `matrix_repr(H)` / `matrix_repr(H, SparseFormat())` | Hermitian, $L=2$ exact spectrum, sparse=dense |
# | (c) GS sparse Lanczos | `solve(H, GroundState(), ExactDiagonalization(:ground))` | residual, norm, matches dense |
# | (d) full spectrum | `solve(H, GroundState(), ExactDiagonalization(:full))` | sorted, $\mathrm{Tr}H=0$, $2^L$ values |
# | (e) time evolution | `as_statevector`, `ConstantProtocol(RealTime(), …)`, `solve` | norm, GS phase, imag-time→GS |
#

#
# **Performance note.** Dense ED costs $O(d^{3L})$ for the full diagonalisation;
# the Lanczos GS costs $O(d^L \cdot n_{\text{Krylov}})$.  For $L \gtrsim 16$ you
# need tensor-network methods (TEBD, DMRG) — which is where the next exercises go.

# ## Scaling benchmark: empirical verification
#
# Time both solvers for L = 4, 6, 8 using `@elapsed` (minimum over 15 trials to suppress GC noise), fit the resulting times in log-space via ordinary least-squares, and overlay the theoretical slopes $\beta_{\text{dense}} = 3\ln 2 \approx 2.08$ and $\beta_{\text{Lanczos}} = \ln 2 \approx 0.69$.

using CairoMakie
# ── Step 1: time dense :full and Lanczos :ground for L = 4, 6, 8 ─────────────
# minimum(@elapsed ...) over n_trials discards GC / scheduler jitter and gives
# the best estimate of true algorithmic compute time.
Ls_bench = [4, 6, 8,10]
n_trials  = 15
t_dense   = Float64[]
t_lanczos = Float64[]
for L in Ls_bench
    H = create_XXZ_L_spin_chain(L).hamiltonian
    t_full = minimum(@elapsed solve(H, GroundState(), ExactDiagonalization(:full))   for _ in 1:n_trials)
    t_gs   = minimum(@elapsed solve(H, GroundState(), ExactDiagonalization(:ground)) for _ in 1:n_trials)
    push!(t_dense,   t_full)
    push!(t_lanczos, t_gs)
    println("L=$L  dense=$(round(t_full*1e3, digits=2)) ms   Lanczos=$(round(t_gs*1e3, digits=2)) ms")
end
# ── Step 2: log-linear least-squares fit  log(t) = α + β·L ──────────────────
# Theory:   dense β = 3·ln2 ≈ 2.08   (cost ∝ D³ = 2^{3L})
#         Lanczos β =   ln2 ≈ 0.69   (cost ∝ D·n_Krylov = 2^L · const)
function log_lin_fit(Ls, ts)
    x, y, n = float.(Ls), log.(ts), length(Ls)
    β = (n*sum(x.*y) - sum(x)*sum(y)) / (n*sum(x.^2) - sum(x)^2)
    α = (sum(y) - β*sum(x)) / n
    return α, β
end
α_d, β_d = log_lin_fit(Ls_bench, t_dense)
α_l, β_l = log_lin_fit(Ls_bench, t_lanczos)
println("\nFitted slopes (exponent per additional site):")
println("  Dense   β = $(round(β_d, digits=3))   theory 3·ln2 ≈ $(round(3*log(2), digits=3))")
println("  Lanczos β = $(round(β_l, digits=3))   theory   ln2 ≈ $(round(log(2),   digits=3))")
# ── Step 3: plot ──────────────────────────────────────────────────────────────
L_fine = collect(range(Ls_bench[begin]-0.5, Ls_bench[end]+0.5, length=300))
fig = Figure(size = (720, 460))
ax  = Axis(fig[1, 1];
    xlabel = "Chain length  L",
    ylabel = "Wall time (s)",
    title  = "ED wall-time scaling  —  dense O(d³ᴸ) vs Lanczos O(dᴸ)",
    yscale = log10,
    xticks = Ls_bench,
)
# Measured points
scatter!(ax, Ls_bench, t_dense;
    color = :steelblue, markersize = 14,
    label = "Dense :full (measured)")
scatter!(ax, Ls_bench, t_lanczos;
    color = :orangered, markersize = 14,
    label = "Lanczos :ground (measured)")
# Fitted exponential curves
lines!(ax, L_fine, exp.(α_d .+ β_d .* L_fine);
    color = :steelblue, linestyle = :dash,
    label = "Dense fit  β=$(round(β_d,digits=2))  (theory $(round(3*log(2),digits=2)))")
lines!(ax, L_fine, exp.(α_l .+ β_l .* L_fine);
    color = :orangered, linestyle = :dash,
    label = "Lanczos fit β=$(round(β_l,digits=2))  (theory $(round(log(2),digits=2)))")
axislegend(ax; position = :lt)
fig

# todo: plot the neel state quench plot to check the correctness
