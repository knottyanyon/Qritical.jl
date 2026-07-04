using LinearAlgebra, Serialization, Qritical

ψ_raw = deserialize(joinpath(DATA_ROOT, "psi.jls"))
N = ndims(ψ_raw) # number of sites
d = size(ψ_raw, 1) # local state space dimension
sites = Tuple([upper(Symbol(:s, i), d) for i in 1:N]) # create the required indices
ψ_tens = QTensor(ψ_raw, sites);
println("State shape: ", size(ψ_raw), "   N = $N   d = $d")

# # Ex 2. MPS Canonical Forms

# ## (a) Left-canonical form
#
#
# `to_mps` performs SVD iteratively left-to-right, absorbing $\Sigma V^\dagger$ rightward at each step. 
# Every tensor $A_i$ satisfies $A_i^\dagger A_i = \mathbb{1}_{\chi_i}$.

# @ the question asked to write the function that does this change from an arbitrary state to left canonical. so write a few lines explain what the `to_mps` function does explaining the various arguments it takes as input and what they mean, especially what the argument form means and add a link directing to the api doc page of the function so that the reader can learn more
# @ probably a bug: in the `to_mps`source` code it says that "- `ψ::QTensor`: full state tensor with ``L`` physical legs, all of type `Lower`". but i want to repeat again that we are talking about the component coefficients of the wavefunction written as a sum in the basis, arranged as the tensor. that means when we start the site indices on `ψ` should all be up indices. not down indices. it is the coefficient matrix that we rearrange into a chain of 3 leg tensors. so please double check this and fix it.  
# to_mps: left-canonical sweep from full state tensor
mps_L = to_mps(ψ_tens; trunc=MaxBondDimTrunc(64), form=:left) # @ add clear explanation of what this MaxBondDimTrunc strategy means.
println("Form  : ", mps_L.form)
println("⟨ψ|ψ⟩ : ", round(real(overlap(mps_L, mps_L)); sigdigits=8)) # @ add a few lines explaining why we choose 8 as the significant digits

# @ explain why it is necessary to do such a verification at each site and how it can help us avoid errors in the long run, the cost of doing such a verification at each site and whether it should be always done
# Verify left-isometry A†A = I at every site
println("Left-isometry errors ‖A†A − I‖:")
for (i, t) in enumerate(mps_L.tensors)
    χL, d_i, χR = size(t.data)
    M   = reshape(t.data, χL * d_i, χR)
    err = norm(M' * M - I(χR))
    println("  site $i: ", round(err; sigdigits=4))
end

# Bond dimension profile
χs = [size(t.data, 3) for t in mps_L.tensors]
println("Bond dims: 1 → ", join(χs, " → "))

# ## (b) Right-canonical form
#
# The right-to-left sweep stores $V^\dagger$ at each site.
# Every tensor $B_i$ satisfies $B_i B_i^\dagger = \mathbb{1}_{\chi_{i-1}}$.

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

# Move the center to different positions and check ⟨ψ|ψ⟩ is preserved
println("\nOverlap ⟨ψ|ψ⟩ for different center positions:")
for l_try in [1, 3, N÷2, N]
    mps_c = canonicalize(mps_L, BondCanonical(l_try))
    println("  l=$l_try: ", round(real(overlap(mps_c, mps_c)); sigdigits=8))
end

# Effect of truncation on accuracy
println("\nTruncation study (D ← bond dimension cap):")
for D in [1, 2, 4, 8, 16, 32]
    mps_trunc = to_mps(ψ_tens; trunc=MaxBondDimTrunc(D), form=:left)
    mps_exact = mps_L                           # D=64 (exact for this state)
    ovlp = real(overlap(mps_trunc, mps_exact))  # ≈ ⟨ψ_D | ψ_exact⟩
    println("  D=$D:  ⟨ψ_D|ψ_exact⟩ = ", round(ovlp; sigdigits=5))
end
