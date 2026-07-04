# # Task 2.3 — Mixed Canonical State

# !!! question "Task 2.3 — Mixed Canonical State"
#     Write a function that performs a mixed canonical decomposition of the state
#     `psi.jls` of [Exercise 1](@ref "Task 1.3 — SVD a state").  Allow for an
#     arbitrary site ``l`` where the state changes from left to right normalized.
#     The function should allow for a maximum matrix dimension ``D`` to truncate
#     the state after each SVD.

using Serialization, LinearAlgebra, Qritical
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI = normpath(joinpath(DATA_ROOT, "psi.jls"))

ψ = deserialize(FPATH_PSI)
L = ndims(ψ)
d = size(ψ, 1)
#--

# ## Mixed canonical form
#
# A **mixed canonical MPS** with *orthogonality center* at site ``l`` combines
# the gauges of Tasks 2.1 and 2.2:
#
# ```math
# \underbrace{A_1 \cdots A_{l-1}}_{\text{left-canonical}} \;
# \underbrace{C_l}_{\text{center}} \;
# \underbrace{B_{l+1} \cdots B_L}_{\text{right-canonical}}
# ```
#
# where
# - ``A_i^\dagger A_i = \mathbb{1}`` (left-isometry) for ``i < l``,
# - ``B_i B_i^\dagger = \mathbb{1}`` (right-isometry) for ``i > l``,
# - ``C_l`` is unconstrained — it carries the full norm of the state.
#
# Mixed canonical form is the natural starting point for computing local
# expectation values: at the center site all environments trivially collapse
# to identity matrices, so ``\langle \psi | O_l | \psi \rangle = \langle C_l | O_l | C_l \rangle``.
#
# **Construction strategy:**
#
# 1. Run the full left-canonical sweep from Task 2.1 to get left-canonical
#    tensors ``A_1, \ldots, A_L``.
# 2. Right-sweep **only** from site ``L`` down to ``l+1``: each step
#    right-canonicalizes site ``i`` and absorbs the leftward factor ``U S``
#    into site ``i-1``.  After ``L - l`` such steps, site ``l`` has absorbed
#    all the right-facing factors and becomes the orthogonality center.

function left_canonical_mps(ψ::Array, D::Int)
    L = ndims(ψ); dims = size(ψ); T = eltype(ψ); RT = real(T)
    tensors  = Vector{Array{T, 3}}(undef, L)
    bond_svs = Vector{Vector{RT}}(undef, L + 1)
    bond_svs[1] = RT[1]; bond_svs[L + 1] = RT[1]
    χL = 1; current = reshape(ψ, 1, :)
    for i in 1:(L-1)
        M = reshape(current, χL * dims[i], :)
        F = svd(M; full=false)
        r = max(min(count(>(0), F.S), D), 1)
        tensors[i] = reshape(F.U[:, 1:r], χL, dims[i], r)
        bond_svs[i + 1] = F.S[1:r]
        current = Diagonal(F.S[1:r]) * F.Vt[1:r, :]
        χL = r
    end
    tensors[L] = reshape(current, χL, dims[L], 1)
    return tensors, bond_svs
end

function mixed_canonical_mps(ψ::Array, l::Int, D::Int)
    L    = ndims(ψ)
    dims = size(ψ)

    tensors, bond_svs = left_canonical_mps(ψ, D)   # full left sweep

    for i in L:-1:(l+1)
        data      = tensors[i]
        χL, d_i, χR = size(data)
        M         = reshape(data, χL, d_i * χR)
        F         = svd(M; full=false)
        r         = max(min(count(>(0), F.S), D), 1)
        tensors[i]  = reshape(F.Vt[1:r, :], r, d_i, χR)
        bond_svs[i] = F.S[1:r]
        L_fac        = F.U[:, 1:r] * Diagonal(F.S[1:r])
        prev         = tensors[i - 1]
        χL_p, d_p, _ = size(prev)
        tensors[i-1] = reshape(reshape(prev, χL_p * d_p, χL) * L_fac, χL_p, d_p, r)
    end

    return tensors, bond_svs
end
#--

# ## Verifying mixed canonical structure
#
# A complete check tests both sides independently and reports the center norm.

function check_mixed_isometry(tensors::Vector{<:Array{<:Number,3}}, l::Int; atol=1e-12)
    all_ok = true

    println("  Left-canonical sites (1 … $(l-1)):")
    for i in 1:(l-1)
        χL, d, χR = size(tensors[i])
        err = norm(reshape(tensors[i], χL * d, χR)' * reshape(tensors[i], χL * d, χR) - I(χR))
        label = err < atol ? "✓" : "✗"
        println("    Site $i ($label):  ‖A†A − I‖ = $(round(err; sigdigits=4))")
        err > atol && (all_ok = false)
    end

    χL_c, d_c, χR_c = size(tensors[l])
    println("  Center site $l:  shape = ($χL_c, $d_c, $χR_c)  ‖C‖ = $(round(norm(tensors[l]); sigdigits=6))")

    println("  Right-canonical sites ($(l+1) … $(length(tensors))):")
    for i in (l+1):length(tensors)
        χL, d, χR = size(tensors[i])
        err = norm(reshape(tensors[i], χL, d * χR) * reshape(tensors[i], χL, d * χR)' - I(χL))
        label = err < atol ? "✓" : "✗"
        println("    Site $i ($label):  ‖BB† − I‖ = $(round(err; sigdigits=4))")
        err > atol && (all_ok = false)
    end

    all_ok || @warn "Mixed canonical verification failed at some sites."
    return all_ok
end
#--

# Decompose `psi.jls` in mixed canonical form with orthogonality center at site ``l = 5``.

D = 64
l = 5
tensors, bond_svs = mixed_canonical_mps(ψ, l, D)

println("Mixed canonical MPS (D = $D, center = site $l):")
check_mixed_isometry(tensors, l)
#--

# ## Moving the center
#
# Once in mixed canonical form, moving the center one site to the right requires
# a single left-to-right SVD on site ``l`` — no global re-sweep is needed.
# This ``O(\chi^3)``-per-step cost is why mixed canonical form is the foundation
# of all DMRG and TEBD sweep algorithms.
#
# Try different center positions and verify the structure each time.

for l_try in [1, 3, L÷2, L]
    ts, _ = mixed_canonical_mps(ψ, l_try, D)
    println("  l = $l_try:  center shape = $(size(ts[l_try]))   ‖C‖ = $(round(norm(ts[l_try]); sigdigits=5))")
end
#--

# ## Qritical.jl equivalent
#
# `move_center!(mps, l)` shifts the orthogonality center of an already-canonical
# `FiniteMPS` to any target site in one call.  After `left_canonical_sweep!` the
# state is in `CanonicalForm(L, L+1)`; `move_center!` right-sweeps exactly as
# many steps as needed to land at site `l`.

mps = FiniteMPS(Spin{1//2}(), L, D)
left_canonical_sweep!(mps)

println("\nBefore move_center!: ", mps.form)

move_center!(mps, l)

println("After  move_center! (l=$l): ", mps.form)
println("  ⟨ψ|ψ⟩ preserved: ", round(real(overlap(mps, mps)); sigdigits=8))
println("  Entanglement entropy at bond $l : ",
        round(entanglement_entropy(mps, l); sigdigits=6), " nats")
#--
