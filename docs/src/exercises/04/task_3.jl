# # Task 4.4 — Adding MPS

# !!! question "Task 4.4 — Adding MPS"
#     Write a function that receives two MPSs ``(|\Psi_1\rangle, |\Psi_2\rangle)``
#     and two weights ``(a, b)`` and returns their weighted sum
#     ``a|\Psi_1\rangle + b|\Psi_2\rangle`` as an MPS.

using Serialization, LinearAlgebra, Qritical
#--

DATA_ROOT  = normpath(joinpath(@__FILE__, ".."))
ψ1_raw = deserialize(normpath(joinpath(DATA_ROOT, "psi1.jls")))
ψ2_raw = deserialize(normpath(joinpath(DATA_ROOT, "psi2.jls")))
L = ndims(ψ1_raw)
#--

# ## Block-diagonal concatenation
#
# Given two MPS with bond dimensions ``\chi_\psi`` and ``\chi_\phi``, their
# sum ``|\Psi\rangle = a|\psi\rangle + b|\phi\rangle`` can be represented
# exactly as an MPS with bond dimension ``\chi_\psi + \chi_\phi``.
# The site tensors are assembled by **block-diagonal concatenation**:
#
# ```math
# W^{[i]}_{\sigma} = \begin{pmatrix} a^{\delta_{i1}} A^{[i]}_\sigma & 0 \\ 0 & b^{\delta_{i1}} B^{[i]}_\sigma \end{pmatrix}
# ```
#
# The weight ``a`` is absorbed into the first site of ``|\psi\rangle`` and
# ``b`` into the first site of ``|\phi\rangle``.  Interior sites just stack
# the two tensors block-diagonally.  At the right boundary the single-column
# structure is restored by vertical concatenation.
#
# Schematically for three sites:
#
# ```
# site 1: cat(a·A¹, b·B¹; dims=3)       shape (1, d, χ_ψ + χ_φ)
# site 2: block_diag(A², B²)             shape (χ_ψ + χ_φ, d, χ_ψ + χ_φ)
# site 3: cat(A³, B³; dims=1)            shape (χ_ψ + χ_φ, d, 1)
# ```

function mps_sum(tensors1::Vector{<:Array{<:Number,3}},
                 tensors2::Vector{<:Array{<:Number,3}},
                 a::Number, b::Number)
    L  = length(tensors1)
    T  = promote_type(eltype(tensors1[1]), eltype(tensors2[1]), typeof(a), typeof(b))
    result = Vector{Array{T, 3}}(undef, L)

    ## TODO: Assemble each site tensor.
    ##
    ## For i in 1:L:
    ##   A = T.(tensors1[i]);  χL_A, d, χR_A = size(A)
    ##   B = T.(tensors2[i]);  χL_B, _, χR_B = size(B)
    ##
    ##   if i == 1
    ##       result[i] = cat(a .* A, b .* B; dims=3)   # (1, d, χR_A + χR_B)
    ##   elseif i == L
    ##       result[i] = cat(A, B; dims=1)              # (χL_A + χL_B, d, 1)
    ##   else
    ##       new = zeros(T, χL_A + χL_B, d, χR_A + χR_B)
    ##       new[1:χL_A,           :, 1:χR_A]           = A
    ##       new[χL_A+1:end,       :, χR_A+1:end]       = B
    ##       result[i] = new
    ##   end

    return result
end
#--

mps1 = FiniteMPS(Spin{1//2}(), L, 16)
mps2 = FiniteMPS(Spin{1//2}(), L, 16)
left_canonical_sweep!(mps1)
left_canonical_sweep!(mps2)

t1 = [t.data for t in mps1.tensors]
t2 = [t.data for t in mps2.tensors]

a, b = 0.6, 0.8
t_sum = mps_sum(t1, t2, a, b)

println("Bond dimensions of sum: ", [size(t, 3) for t in t_sum])
#--

# ## Verify linearity via overlap
#
# ``\langle\psi_1 | a\psi_1 + b\psi_2\rangle = a\langle\psi_1|\psi_1\rangle
# + b\langle\psi_1|\psi_2\rangle``.

ovlp_11 = overlap(mps1, mps1)
ovlp_12 = overlap(mps1, mps2)
expected = a * ovlp_11 + b * ovlp_12

println("Expected ⟨ψ₁|aψ₁+bψ₂⟩ = ", round(real(expected); sigdigits=6))
#--

# ## Qritical.jl equivalent
#
# `a * ψ + b * φ` is directly supported via `Base.:+` and scalar
# multiplication on `FiniteMPS`:

mps_sum_qritical = a * mps1 + b * mps2
println("Qritical ⟨ψ₁|aψ₁+bψ₂⟩ = ",
        round(real(overlap(mps1, mps_sum_qritical)); sigdigits=6))
#--
