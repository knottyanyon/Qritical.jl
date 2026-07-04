# # Task 4.3 — Observables

# !!! question "Task 4.3 — Observables"
#     Write a function that receives an MPS in mixed canonical form and
#     evaluates expectation values of ``\sigma^z`` and ``\sigma^x`` efficiently
#     at the site where the normalization switches from left to right.

using Serialization, LinearAlgebra, Qritical
#--

DATA_ROOT  = normpath(joinpath(@__FILE__, ".."))
ψ_raw = deserialize(normpath(joinpath(DATA_ROOT, "psi1.jls")))
L = ndims(ψ_raw)
#--

# ## Why mixed canonical form makes observables cheap
#
# In a general MPS, evaluating ``\langle\psi|O_l|\psi\rangle`` at site ``l``
# requires contracting two full environments (left boundary → site ``l`` and
# site ``l`` → right boundary): cost ``O(L\chi^3)``.
#
# In **mixed canonical form** with orthogonality center at site ``l``, both
# environments collapse to identity matrices.  All tensors to the left satisfy
# ``A_i^\dagger A_i = \mathbb{1}`` and all tensors to the right satisfy
# ``B_i B_i^\dagger = \mathbb{1}``, so:
#
# ```math
# \langle\psi | O_l | \psi \rangle
# = \sum_{\sigma,\sigma'} (C_l)^*_{\alpha\sigma\beta} \,
#   O^{\sigma'}_\sigma \, (C_l)_{\alpha\sigma'\beta}
# = \operatorname{tr}(C_l^\dagger \, \bar{O}_l \, C_l)
# ```
#
# where the trace runs over the virtual legs ``\alpha, \beta`` and
# ``C_l`` is the center tensor.  Cost: ``O(\chi^2 d^2)`` — independent of ``L``!

# Spin-1/2 operators in the ``\{|\uparrow\rangle, |\downarrow\rangle\}`` basis:
σz = [1.0  0.0; 0.0 -1.0]
σx = [0.0  1.0; 1.0  0.0]
#--

function local_observable(tensors::Vector{<:Array{<:Number,3}},
                           op::Matrix{<:Number},
                           l::Int)
    C = tensors[l]                              # (χL, d, χR)
    χL, d, χR = size(C)
    C_mat = reshape(permutedims(C, (1,3,2)), χL * χR, d)   # (χL·χR, d)
    ρ = C_mat' * C_mat                          # (d, d) reduced density matrix
    return real(tr(op * ρ)) / real(tr(ρ))
end
#--

mps = FiniteMPS(Spin{1//2}(), L, 32)
left_canonical_sweep!(mps)

tensors = [t.data for t in mps.tensors]

println("Expectation values at each orthogonality center:")
println("  site  |  ⟨σᶻ⟩     |  ⟨σˣ⟩")
for l in 1:L
    mps_l = deepcopy(mps)
    move_center!(mps_l, l)
    ts = [t.data for t in mps_l.tensors]
    sz = local_observable(ts, σz, l)
    sx = local_observable(ts, σx, l)
    println("  $l     |  $(round(sz; sigdigits=4))  |  $(round(sx; sigdigits=4))")
end
#--

# ## Verify: sum of ⟨σᶻ⟩ should be consistent across sites
#
# For a fixed MPS the expectation value ``\langle\sigma^z_l\rangle`` is a
# physical property — it must not change depending on which site we chose as
# the orthogonality center when we computed it.  (Canonical form is a gauge
# choice; it cannot affect physical observables.)

println("\nConsistency check: ⟨σᶻ₁⟩ computed with center at every site:")
for l_center in 1:L
    mps_c = deepcopy(mps)
    move_center!(mps_c, l_center)
    ts = [t.data for t in mps_c.tensors]
    println("  center=$l_center → ⟨σᶻ₁⟩ = ", round(local_observable(ts, σz, 1); sigdigits=6))
end
#--
