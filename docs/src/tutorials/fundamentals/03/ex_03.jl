using LinearAlgebra, Serialization, Qritical
const DATA_ROOT = normpath(joinpath(@__FILE__, "..", "..", "..", "data"))
ψ_raw = deserialize(joinpath(DATA_ROOT, "psi.jls"))
N = ndims(ψ_raw);  d = size(ψ_raw, 1)
sites  = Tuple([upper(Symbol(:s, i), d) for i in 1:N])
ψ_tens = QTensor(ψ_raw, sites)
# Build a reference left-canonical MPS (exact, D=64)
mps_ref = to_mps(ψ_tens; trunc=MaxBondDimTrunc(64), form=:left)
println("Reference MPS form: ", mps_ref.form)

# # Ex 3. Transformations Between Canonical Forms
#
# **Week 3 — where this sits in the arc.** Week 2 built canonical MPS *from a full
# state vector*. This week we never touch a full state vector again: we convert a
# canonical MPS **directly into another canonical form**, tensor by tensor. That
# is the whole point — the exercise sheet asks for the conversions *"without
# recovering the full wave-function"*, because rebuilding the $d^{N}$ vector would
# throw away the exponential advantage the MPS bought us in the first place.
#
# All transformations sweep along the chain applying one SVD per site — cost $O(L\chi^3)$,
# no reconstruction of the full $d^L$ state vector.
#
# !!! info "The one mechanism behind every canonical form: SVD makes isometries"
#     Reshape a site tensor so its physical leg groups with **one** bond, SVD it,
#     keep the isometric factor ($U$ or $V^\dagger$) as the new canonical tensor,
#     and push the leftover ($SV^\dagger$ or $US$) into the neighbour. Sweeping
#     this operation along the chain drags the "non-canonical remainder" — and with
#     it the norm — from one end to the other. Left- vs right-canonical differ only
#     in *which* bond the physical leg groups with and *which way* the remainder is
#     pushed (Schollwöck §4.4.1–4.4.2).
#
# !!! note "SVD or QR? Both gauge-transform; only SVD can truncate"
#     For a *pure* gauge change the singular values are not needed, so a QR ($A=Q$)
#     or LQ ($B=Q^\dagger$) decomposition is the cheaper way to produce the
#     isometry. But QR is **not rank-revealing** — it cannot tell you which bond
#     states are negligible — so it keeps the full bond dimension. The moment you
#     want to *truncate* (as this exercise does, capping each bond at $D$), you need
#     the singular-value spectrum and must use SVD.

# ## (a) Left → Right canonical
#
# Apply `canonicalize(mps, RightCanonical())` to an already left-canonical MPS.
# The sweep goes site $L \to 1$; at each step the $V^\dagger$ factor becomes the
# new right-canonical tensor and the $U\Sigma$ factor is absorbed leftward.

mps_R = canonicalize(mps_ref, RightCanonical())
println("After L→R sweep: ", mps_R.form)
println("⟨ψ|ψ⟩ preserved: ", round(real(overlap(mps_R, mps_R)); sigdigits=8))

# The overlap is our proof that the conversion was a faithful *gauge* move: a
# right-canonical $B$-tensor satisfies $BB^\dagger=\mathbb{1}$, so we verify that
# property at every site. Any site that fails would mean the sweep left a
# non-isometric remainder behind.
# Verify: every site satisfies BB† = I  (note the parens — `let (a,b,c) = …`)
errs = [let (χL, d_i, χR) = size(t.data)
            M = reshape(t.data, χL, d_i * χR)
            norm(M * M' - I(χL))
        end for t in mps_R.tensors]
println("Right-isometry errors: ", round.(errs; sigdigits=3))

# The Schmidt spectrum on a bond is a *physical* property of the state, so it must
# survive the gauge change untouched — the left- and right-canonical numbers below
# should agree digit for digit.
# Singular values must be gauge-invariant
println("Bond 5 SVs (left canonical):  ", round.(mps_ref.bond_svs[6].values; sigdigits=4))
println("Bond 5 SVs (right canonical): ", round.(mps_R.bond_svs[6].values;   sigdigits=4))

# ## (b) Right → Left canonical
#
# Reverse direction: sweep site $1 \to L$, storing the $U$ factor as the new
# left-canonical tensor and absorbing $\Sigma V^\dagger$ rightward.
#
# Converting $L\!\to\!R$ and then $R\!\to\!L$ is a **round trip**: starting from a
# left-canonical state and coming back to one should return the same physical
# state, so the overlap below is again our correctness check. (With truncation
# switched on, the round trip is *not* exactly reversible — the discarded weight
# from the first sweep is gone for good; see the note on one-sided interdependence.)

mps_R2L = canonicalize(mps_R, LeftCanonical())
println("After R→L sweep: ", mps_R2L.form)
println("⟨ψ|ψ⟩ preserved: ", round(real(overlap(mps_R2L, mps_R2L)); sigdigits=8))

# Verify left-isometry everywhere
errs2 = [let (χL, d_i, χR) = size(t.data)
             M = reshape(t.data, χL * d_i, χR)
             norm(M' * M - I(χR))
         end for t in mps_R2L.tensors]
println("Left-isometry errors: ", round.(errs2; sigdigits=3))

# !!! warning "Truncation while sweeping has a direction bias"
#     Keeping the $D$ largest singular values at each bond is the *optimal* rank-$D$
#     approximation of that one matrix in Frobenius norm, with discarded weight
#     $\varepsilon = \sum_{\alpha>D}\lambda_\alpha^2$. But each site inherits the
#     already-truncated factor from the previous step, so **truncations on one side
#     depend on those on the other, not vice versa** (Schollwöck §4.5.1). This is
#     harmless for small $\varepsilon$ and mildly suboptimal for aggressive cuts —
#     and it is exactly the truncation that TEBD and tDMRG perform while sweeping.

# ## (c) Normalization diagnostics
#
# `canonical_error(A)` measures the left-isometry deviation $\|A^\dagger A - I\|_F$.
# For a fully left-canonical MPS all errors should be $\approx 0$.
# For a right-canonical MPS the left errors should be large and right errors $\approx 0$.
# `is_canonical` returns true if all errors are below a tolerance.
#
# !!! info "Why the Frobenius norm of $(\text{gram}-\mathbb{1})$ is the right diagnostic"
#     Part 3 of the sheet asks us to *devise a distance-from-unity measure*. The
#     natural choice is $\eta_i = \lVert G_i - \mathbb{1}\rVert_F$ where
#     $G_i=\sum_\sigma A_i^{\sigma\dagger}A_i^{\sigma}$ is the site's Gram matrix.
#     It is the standard scalar diagnostic because it is **basis-independent**
#     (unitarily invariant), it is **exactly zero iff** the tensor is canonical,
#     and it **grows smoothly** as the tensor is perturbed away from isometry —
#     so it doubles as a convergence monitor for iterative algorithms. The table
#     below prints both the left ($\eta^{(A)}$) and right ($\eta^{(B)}$) deviation
#     per site, which makes the *location of the orthogonality centre* visible at
#     a glance: it is the one site where neither column is small.

function norm_table(mps, label)
    println("\n$label  (form = $(mps.form))")
    println("  site │ δ_L (left iso) │ δ_R (right iso)")
    println("  ─────┼────────────────┼────────────────")
    for (i, t) in enumerate(mps.tensors)
        χL, d_i, χR = size(t.data)
        ML  = reshape(t.data, χL * d_i, χR)
        MR  = reshape(t.data, χL, d_i * χR)
        δ_L = norm(ML' * ML - I(χR))
        δ_R = norm(MR * MR' - I(χL))
        println("  $(lpad(i,4)) │ $(rpad(round(δ_L; sigdigits=3), 15)) │ $(round(δ_R; sigdigits=3))")
    end
end
norm_table(mps_ref, "Left-canonical")

norm_table(mps_R, "Right-canonical")

# Mixed canonical: sites 1..l left-iso, site l+1..N right-iso
l = N ÷ 2
mps_M = canonicalize(mps_ref, BondCanonical(l))
norm_table(mps_M, "Mixed canonical (center bond = $l)")

# `is_canonical` collapses the whole table into a single yes/no by testing every
# $\eta_i$ against a tolerance — the cheap flag you call in production, with the
# full table reserved for when it returns `false` and you need to see *where*.
# is_canonical as a quick flag
println("is_canonical(mps_L): ", is_canonical(mps_ref))
println("is_canonical(mps_R): ", is_canonical(mps_R))
println("is_canonical(mps_M): ", is_canonical(mps_M))

# All three forms describe the same state, so the Schmidt spectrum at a given bond
# is one more gauge invariant they must share — a final end-to-end consistency check.
# Gauge invariance of singular-value spectra across all three forms
println("\nSingular values at bond ", N÷2, ":")
for (lbl, mps) in [("Left", mps_ref), ("Right", mps_R), ("Mixed", mps_M)]
    svs = round.(mps.bond_svs[N÷2 + 1].values; sigdigits=4)
    println("  $lbl:  ", svs)
end

# ## (d) XXZ spin-½ ↔ spinless fermions (Jordan–Wigner)
#
# Part 4 of the sheet is pen-and-paper: map the spin-½ XXZ chain onto spinless
# fermions. There is no tensor code here, but the mapping is the bridge to the
# fermionic models of Week 6+, so it is worth recording alongside the solution.
#
# !!! note "The Jordan–Wigner dictionary"
#     Attach a nonlocal **string** to each spin so that operators on different
#     sites anticommute as fermions must:
#     ```math
#     S^{+}_j = f^{\dagger}_j \exp\!\Big(i\pi\!\!\sum_{k<j} n_k\Big),\qquad
#     S^{-}_j = \exp\!\Big(i\pi\!\!\sum_{k<j} n_k\Big) f_j,\qquad
#     S^{z}_j = n_j - \tfrac12,\quad n_j=f^{\dagger}_j f_j .
#     ```
#     In a **nearest-neighbour** product the strings between $j$ and $j{+}1$
#     cancel, so the flip term becomes a clean hopping and $S^z S^z$ becomes a
#     density–density interaction. The XXZ Hamiltonian
#     ```math
#     H = \sum_j \Big[\tfrac{J}{2}\big(S^{+}_j S^{-}_{j+1}+\text{h.c.}\big)
#                     + J\,\Delta\, S^{z}_j S^{z}_{j+1}\Big]
#     ```
#     maps to
#     ```math
#     H = \sum_j \Big[\tfrac{J}{2}\big(f^{\dagger}_j f_{j+1}+\text{h.c.}\big)
#             + J\Delta\big(n_j-\tfrac12\big)\big(n_{j+1}-\tfrac12\big)\Big],
#     ```
#     i.e. free hopping at the isotropic point $\Delta=0$ (the XX chain is exactly
#     solvable free fermions), with $\Delta$ tuning the nearest-neighbour
#     repulsion. This is why the XX limit is the standard analytic benchmark for
#     the time-evolution exercises later on.
