# # Task 3.1 — From Left to Right Canonical

# !!! question "Task 3.1 — From Left to Right"
#     Write a function that takes a left canonical representation of `psi.jls`
#     and transforms it into a right canonical one *without* recovering the full
#     wave-function as an intermediate step.  The function should allow for a
#     maximum matrix dimension ``D`` to truncate the state after each SVD.

using Serialization, LinearAlgebra, Qritical
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))

ψ = deserialize(normpath(joinpath(DATA_ROOT, "psi.jls")))
L = ndims(ψ)
d = size(ψ, 1)
#--

# ## Sweeping in MPS space
#
# In Exercise 2 we built an MPS by decomposing the full ``d^L``-dimensional
# state tensor — a cost that grows exponentially in ``L``.  Once we *have* an
# MPS, every subsequent canonical transformation can be done by sweeping along
# the chain, applying one SVD per site.  Memory cost: ``O(L \chi^2 d)``.
#
# **Left → right canonical** sweeps from site ``L`` down to site ``1``.
# At each step the current tensor is reshaped into a ``\chi_L \times (d\chi_R)``
# matrix, thin-SVD'd, and the right factor ``V^\dagger`` becomes the new
# right-canonical tensor ``B_i``.  The leftward factor ``U\Sigma`` is absorbed
# into site ``i-1``, which will be processed on the next iteration:
#
# ```math
# M_i = U_i \Sigma_i V_i^\dagger, \qquad
# B_i = \operatorname{reshape}(V_i^\dagger[1{:}r,:],\, r, d, \chi_R), \qquad
# A_{i-1} \leftarrow A_{i-1} \cdot (U_i \Sigma_i)
# ```

function left_to_right!(tensors::Vector{<:Array{<:Number,3}},
                         bond_svs::Vector{<:Vector{<:Real}},
                         D::Int)
    L = length(tensors)

    ## TODO: Sweep from site L down to site 1.
    ##
    ## For i in L:-1:1:
    ##   1. data = tensors[i];  χL, d_i, χR = size(data)
    ##   2. M    = reshape(data, χL, d_i * χR)
    ##   3. F    = svd(M; full=false)
    ##      r    = min(count(>(0), F.S), D)
    ##   4. tensors[i]  = reshape(F.Vt[1:r, :], r, d_i, χR)
    ##      bond_svs[i] = F.S[1:r]
    ##   5. if i > 1
    ##          L_fac        = F.U[:, 1:r] * Diagonal(F.S[1:r])
    ##          prev         = tensors[i - 1]
    ##          χL_p, d_p, _ = size(prev)
    ##          tensors[i-1] = reshape(reshape(prev, χL_p * d_p, χL) * L_fac,
    ##                                 χL_p, d_p, r)
    ##      end

    return tensors, bond_svs
end
#--

# Start from a left-canonical MPS built via the sweep approach.

mps = FiniteMPS(Spin{1//2}(), L, 64)
left_canonical_sweep!(mps)

tensors  = [copy(t.data) for t in mps.tensors]
bond_svs = deepcopy(mps.bond_svs)

D = 64
left_to_right!(tensors, bond_svs, D)
#--

# ## Verify: ``B_i B_i^\dagger = \mathbb{1}`` at every site

function check_right_iso(tensors; atol=1e-12)
    all_ok = true
    for (i, B) in enumerate(tensors)
        χL, d, χR = size(B)
        err = norm(reshape(B, χL, d * χR) * reshape(B, χL, d * χR)' - I(χL))
        label = err < atol ? "✓" : "✗"
        println("  Site $i ($label):  ‖BB† − I‖ = $(round(err; sigdigits=4))")
        err > atol && (all_ok = false)
    end
    return all_ok
end

check_right_iso(tensors)
#--

# ## Qritical.jl equivalent
#
# `right_canonical_sweep!(mps)` performs this same sweep and updates
# `mps.form` to `CanonicalForm(0, 1)`.

mps2 = deepcopy(mps)
right_canonical_sweep!(mps2)
println("Form: ", mps2.form, "   ⟨ψ|ψ⟩ = ", round(real(overlap(mps2, mps2)); sigdigits=8))
#--
