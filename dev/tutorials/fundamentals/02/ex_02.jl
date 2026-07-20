using LinearAlgebra, Serialization, Qritical

const DATA_ROOT = normpath(joinpath(@__FILE__, "..", "..", "..", "data"))

ψ_raw = deserialize(joinpath(DATA_ROOT, "psi.jls"))
N = ndims(ψ_raw) # number of sites
d = size(ψ_raw, 1) # local state space dimension
sites = Tuple([upper(Symbol(:s, i), d) for i in 1:N]) # create the required indices
ψ_tens = QTensor(ψ_raw, sites);
println("State shape: ", size(ψ_raw), "   N = $N   d = $d")

# # Ex 2. MPS Canonical Forms
#
# **Week 2 — where this sits in the arc.** In Week 1 we took a *single* matrix
# apart with one SVD and read off its Schmidt rank. This week we iterate that
# idea across an entire lattice. A general pure state on $N$ sites,
#
# ```math
# |\psi\rangle = \sum_{\sigma_1\ldots\sigma_N} c_{\sigma_1\ldots\sigma_N}\,
#                 |\sigma_1,\ldots,\sigma_N\rangle ,
# ```
#
# is a single tensor with $d^{N}$ coefficients — exponentially large and
# completely non-local. A **matrix product state (MPS)** re-expresses *the very
# same coefficients* as a chain of three-leg tensors,
#
# ```math
# c_{\sigma_1\ldots\sigma_N} = A^{\sigma_1}A^{\sigma_2}\cdots A^{\sigma_N},
# ```
# by a cascade of SVDs (Schollwöck §4.1.3). Nothing is approximated yet: this is
# an *exact* rewriting. What we gain is a **gauge freedom** — the same physical
# state admits many equivalent tensor chains — and this exercise is about the
# three gauge choices that make the tensors *isometries*: **left-canonical**,
# **right-canonical**, and **mixed-canonical**.
#
# !!! info "Why bother making the tensors isometric?"
#     A canonical form is not cosmetic. When each tensor is an isometry, the
#     environment blocks it contracts against collapse to the identity, so
#     norms, overlaps and local expectation values become *cheap and numerically
#     stable* to evaluate (Weeks 4–5), and — crucially — the singular values
#     sitting on a bond become genuine **Schmidt coefficients**, which is what
#     lets us truncate in a controlled way. Canonical forms are the coordinate
#     system in which every later algorithm (TEBD, DMRG) is written.
#
# !!! note "A note on leg direction"
#     The physical legs of `ψ_tens` are built with `upper(...)`, i.e. they are
#     **ket ("up") indices**. That is deliberate: $c_{\sigma_1\ldots\sigma_N}$
#     are the expansion coefficients of $|\psi\rangle$ in the computational
#     basis, so reshaping them into a tensor keeps every physical leg a ket. The
#     canonicalisation below only ever *regroups* these coefficients into a
#     chain — it never touches their variance.

# ## (a) Left-canonical form
#
# The construction sweeps **left to right**. Reshape the coefficient tensor into
# a matrix $\Psi_{\sigma_1,(\sigma_2\ldots\sigma_N)}$, SVD it as $U\Sigma V^\dagger$,
# keep $U$ as the first site tensor $A_1$, and absorb $\Sigma V^\dagger$ into the
# remaining legs. Repeat on the next site, and so on. Because every $U$ from an
# SVD satisfies $U^\dagger U = \mathbb{1}$, each stored tensor inherits the
# **left-isometry** property
#
# ```math
# \sum_{\sigma_i} A_i^{\sigma_i\dagger} A_i^{\sigma_i}
#     = A_i^\dagger A_i = \mathbb{1}_{\chi_i}.
# ```
#
# Physically, the columns of $A_i$ form an *orthonormal basis for the left block*
# $\{1,\ldots,i\}$ — the isometry condition is orthonormality of those block
# states written index-wise.
#
# !!! info "What `to_mps` does, argument by argument"
#     `to_mps(ψ; trunc, form)` performs exactly the iterated-SVD sweep above:
#       - `ψ::QTensor` — the full state tensor with all $N$ physical (ket) legs;
#       - `form` — which sweep direction/gauge to produce: `:left` sweeps
#         left→right and stores left-isometric $A$-tensors (this part), `:right`
#         sweeps the other way (part b);
#       - `trunc` — the truncation policy applied to the singular values at each
#         bond as they are produced.
#     See the [`to_mps`](@ref) API entry for the full signature and return type.

# !!! note "What `MaxBondDimTrunc(64)` means"
#     `MaxBondDimTrunc(D)` is the simplest truncation policy: at every bond keep
#     **at most `D` singular values** — the `D` largest — and discard the rest.
#     It is a hard cap on the bond dimension $\chi_i \le D$, which bounds the
#     memory ($O(N D^2 d)$) and the cost of every later contraction. Discarding
#     singular values throws away the least-entangled Schmidt components, so in
#     general it introduces a truncation error; here `D = 64` is large enough to
#     retain the *entire* spectrum of this particular state, so the sweep stays
#     **exact** and we can use it as the ground truth in part (c).
# to_mps: left-canonical sweep from full state tensor
mps_L = to_mps(ψ_tens; trunc=MaxBondDimTrunc(64), form=:left)
println("Form  : ", mps_L.form)
# The MPS is normalised, so $\langle\psi|\psi\rangle$ should equal $1$. We print
# it to `sigdigits=8`: full `Float64` carries ~16 significant digits, but a chain
# of $N$ SVDs accumulates rounding error, so agreement to ~8 digits is the honest
# expectation. Showing 8 digits reveals any real deviation while not drowning the
# result in the last few bits of floating-point noise.
println("⟨ψ|ψ⟩ : ", round(real(overlap(mps_L, mps_L)); sigdigits=8))

# **Verifying the isometry at every site.** The left-canonical property is an
# *invariant* the sweep is supposed to maintain — so checking it is our first
# line of defence against bugs. A bad reshape, an off-by-one in the leg grouping,
# or an over-aggressive truncation all show up immediately as $\|A^\dagger A -
# \mathbb{1}\|$ drifting away from zero. Each check costs only $O(\chi^2 d)$ —
# negligible next to building the MPS — which is why it is worth doing per site
# *while developing*; in production one would verify once and then trust the form.
# Verify left-isometry A†A = I at every site
println("Left-isometry errors ‖A†A − I‖:")
for (i, t) in enumerate(mps_L.tensors)
    χL, d_i, χR = size(t.data)
    M   = reshape(t.data, χL * d_i, χR)
    err = norm(M' * M - I(χR))
    println("  site $i: ", round(err; sigdigits=4))
end

# **Bond-dimension profile.** The bond dimensions grow from the edges toward the
# middle — $\chi_i \le \min(d^{\,i}, d^{\,N-i})$ — because the left and right
# blocks can only share as many Schmidt states as the smaller of the two can
# support. The peak in the centre is the maximal entanglement this state carries.
# Bond dimension profile
χs = [size(t.data, 3) for t in mps_L.tensors]
println("Bond dims: 1 → ", join(χs, " → "))

# ## (b) Right-canonical form
#
# Everything mirrors part (a) under left$\leftrightarrow$right reflection. The
# sweep runs **right to left**, storing the $V^\dagger$ factor at each site as a
# $B$-tensor, so the stored tensors are **right-isometric**:
#
# ```math
# \sum_{\sigma_i} B_i^{\sigma_i} B_i^{\sigma_i\dagger}
#     = B_i B_i^\dagger = \mathbb{1}_{\chi_{i-1}} .
# ```
#
# Now it is the *rows* of $B_i$ that are orthonormal, giving an orthonormal basis
# for the right block $\{i,\ldots,N\}$.

mps_R = to_mps(ψ_tens; trunc=MaxBondDimTrunc(64), form=:right)
println("Form  : ", mps_R.form)
println("⟨ψ|ψ⟩ : ", round(real(overlap(mps_R, mps_R)); sigdigits=8))

# Verify right-isometry BB† = I
println("Right-isometry errors ‖BB† − I‖:")
for (i, t) in enumerate(mps_R.tensors)
    χL, d_i, χR = size(t.data)
    M   = reshape(t.data, χL, d_i * χR)
    err = norm(M * M' - I(χL))
    println("  site $i: ", round(err; sigdigits=4))
end

# !!! info "The singular values are gauge-independent"
#     Left- and right-canonical forms are two different *gauges* for the **same**
#     physical state, so any genuinely physical quantity must agree between them.
#     The sharpest such quantity is the **Schmidt spectrum** across a bond: the
#     singular values at a given cut are the coefficients of the state's Schmidt
#     decomposition there, and the Schmidt decomposition is unique (up to
#     degeneracies). The check below prints the spectrum at the central bond in
#     both gauges — they must coincide to numerical precision, confirming that
#     canonicalisation only reshuffled the tensors, not the entanglement.
# Physical invariant: singular values must be gauge-independent
println("Singular values at bond 5 (middle):")
println("  Left  canonical: ", round.(mps_L.bond_svs[6].values; sigdigits=4))
println("  Right canonical: ", round.(mps_R.bond_svs[6].values; sigdigits=4))

# ## (c) Mixed-canonical form
#
# Put the orthogonality centre at site $l$: sites $1\ldots l-1$ are left-canonical,
# site $l$ is unconstrained (carries the full norm), sites $l+1\ldots L$ are right-canonical.
#
# API: `canonicalize(mps, BondCanonical(l))` — returns a new MPS with center AT BOND $l$,
# meaning sites $1\ldots l$ are left-canonical and sites $l+1\ldots L$ are right-canonical.
#
# !!! info "Why the mixed form is the important one"
#     Left- and right-canonical forms push the norm all the way to one edge. The
#     mixed form instead parks it on a chosen bond, and *there* the singular
#     values are literally the Schmidt coefficients $\{\lambda_a\}$ of the
#     bipartition $[1\ldots l\,|\,l{+}1\ldots N]$:
#     ```math
#     |\psi\rangle = \sum_{a} \lambda_a\,|a\rangle_{\!A}\otimes|a\rangle_{\!B},
#     \qquad |a\rangle_A \text{ from the } A\text{'s},\ |a\rangle_B \text{ from the } B\text{'s}.
#     ```
#     This is the exact many-body analogue of the single-matrix SVD from Week 1,
#     and it is the reason the *orthogonality centre* is the natural place to
#     apply gates, read off entanglement entropy, and truncate — everything the
#     later exercises do happens at the centre.
#
# ```
#    A -- A -- A -- A --◆-- B -- B -- B      ◆ = orthogonality centre (bond l)
#    └ left-canonical ┘   └ right-canonical ┘
# ```

l = 5
mps_M = canonicalize(mps_L, BondCanonical(l))
println("Form after BondCanonical($l): ", mps_M.form)

# Verify mixed structure
println("Left-isometry errors (sites 1…$l):")
for i in 1:l
    χL, d_i, χR = size(mps_M.tensors[i].data)
    M   = reshape(mps_M.tensors[i].data, χL * d_i, χR)
    err = norm(M' * M - I(χR))
    println("  site $i (left ): ", round(err; sigdigits=4))
end
println("Right-isometry errors (sites $(l+1)…$N):")
for i in (l+1):N
    χL, d_i, χR = size(mps_M.tensors[i].data)
    M   = reshape(mps_M.tensors[i].data, χL, d_i * χR)
    err = norm(M * M' - I(χL))
    println("  site $i (right): ", round(err; sigdigits=4))
end

# **Moving the centre is a pure gauge move.** Sliding the orthogonality centre to
# a different bond just shifts an isometry across one link and leaves the physical
# state untouched — so $\langle\psi|\psi\rangle$ must be invariant no matter where
# we place it. That invariance is exactly what makes it safe to walk the centre up
# and down the chain during a DMRG or TEBD sweep.
# Move the center to different positions and check ⟨ψ|ψ⟩ is preserved
println("\nOverlap ⟨ψ|ψ⟩ for different center positions:")
for l_try in [1, 3, N÷2, N]
    mps_c = canonicalize(mps_L, BondCanonical(l_try))
    println("  l=$l_try: ", round(real(overlap(mps_c, mps_c)); sigdigits=8))
end

# **How much can we throw away?** Finally we make truncation *bite*: rebuild the
# MPS with an increasingly tight cap $D$ and measure the overlap
# $\langle\psi_D|\psi_{\text{exact}}\rangle$ with the full-rank state. Because the
# discarded weight is set by the tail of the Schmidt spectrum, the overlap climbs
# toward $1$ as fast as those singular values decay — a rapidly decaying spectrum
# means a small $D$ already captures the state, which is precisely the empirical
# fact that makes MPS methods work for physical (low-entanglement) states.
# Effect of truncation on accuracy
println("\nTruncation study (D ← bond dimension cap):")
for D in [1, 2, 4, 8, 16, 32]
    mps_trunc = to_mps(ψ_tens; trunc=MaxBondDimTrunc(D), form=:left)
    mps_exact = mps_L                           # D=64 (exact for this state)
    ovlp = real(overlap(mps_trunc, mps_exact))  # ≈ ⟨ψ_D | ψ_exact⟩
    println("  D=$D:  ⟨ψ_D|ψ_exact⟩ = ", round(ovlp; sigdigits=5))
end
