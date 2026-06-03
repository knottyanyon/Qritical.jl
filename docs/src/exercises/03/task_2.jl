# # Task 3.2 — From Right to Left Canonical

# !!! question "Task 3.2 — From Right to Left"
#     Write a function that takes a right canonical representation of `psi.jls`
#     and transforms it into a left canonical one *without* recovering the full
#     wave-function as an intermediate step.  The function should allow for a
#     maximum matrix dimension ``D`` to truncate the state after each SVD.

using Serialization, LinearAlgebra, Qritical
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))

ψ = deserialize(normpath(joinpath(DATA_ROOT, "psi.jls")))
L = ndims(ψ)
d = size(ψ, 1)
#--

# ## Right → left canonical sweep
#
# This is the mirror image of Task 3.1.  Starting from a right-canonical MPS
# we sweep from site ``1`` up to site ``L``.  At each step the current tensor
# is reshaped into a ``(\chi_L d) \times \chi_R`` matrix, thin-SVD'd, and the
# left factor ``U`` becomes the new left-canonical tensor ``A_i``.  The
# rightward factor ``\Sigma V^\dagger`` is absorbed into site ``i+1``:
#
# ```math
# M_i = U_i \Sigma_i V_i^\dagger, \qquad
# A_i = \operatorname{reshape}(U_i[:,1{:}r],\, \chi_L, d, r), \qquad
# B_{i+1} \leftarrow (\Sigma_i V_i^\dagger) \cdot B_{i+1}
# ```

function right_to_left!(tensors::Vector{<:Array{<:Number,3}},
                         bond_svs::Vector{<:Vector{<:Real}},
                         D::Int)
    L = length(tensors)

    ## TODO: Sweep from site 1 up to site L.
    ##
    ## For i in 1:L:
    ##   1. data = tensors[i];  χL, d_i, χR = size(data)
    ##   2. M    = reshape(data, χL * d_i, χR)
    ##   3. F    = svd(M; full=false)
    ##      r    = min(count(>(0), F.S), D)
    ##   4. tensors[i]      = reshape(F.U[:, 1:r], χL, d_i, r)
    ##      bond_svs[i + 1] = F.S[1:r]
    ##   5. if i < L
    ##          R        = Diagonal(F.S[1:r]) * F.Vt[1:r, :]
    ##          nxt      = tensors[i + 1]
    ##          _, d_n, χRn = size(nxt)
    ##          tensors[i+1] = reshape(R * reshape(nxt, χR, d_n * χRn), r, d_n, χRn)
    ##      end

    return tensors, bond_svs
end
#--

mps = FiniteMPS(Spin{1//2}(), L, 64)
right_canonical_sweep!(mps)

tensors  = [copy(t.data) for t in mps.tensors]
bond_svs = deepcopy(mps.bond_svs)

D = 64
right_to_left!(tensors, bond_svs, D)
#--

# ## Verify: ``A_i^\dagger A_i = \mathbb{1}`` at every site

function check_left_iso(tensors; atol=1e-12)
    all_ok = true
    for (i, A) in enumerate(tensors)
        χL, d, χR = size(A)
        err = norm(reshape(A, χL * d, χR)' * reshape(A, χL * d, χR) - I(χR))
        label = err < atol ? "✓" : "✗"
        println("  Site $i ($label):  ‖A†A − I‖ = $(round(err; sigdigits=4))")
        err > atol && (all_ok = false)
    end
    return all_ok
end

check_left_iso(tensors)
#--

# ## Qritical.jl equivalent

mps2 = deepcopy(mps)
left_canonical_sweep!(mps2)
println("Form: ", mps2.form, "   ⟨ψ|ψ⟩ = ", round(real(overlap(mps2, mps2)); sigdigits=8))
#--
