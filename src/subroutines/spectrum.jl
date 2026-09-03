#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs:
credits: N/A
=#

"""
    SingValSpectrum{V<:AbstractVector{<:Real}}

Singular-value spectrum extracted from one bond's factorization during a canonicalization sweep.

# Fields

$(Glossaries.Field{@__MODULE__}()([:sv_values, :sv_epsilon, :sv_normalized]))
"""
struct SingValSpectrum{V<:AbstractVector{<:Real}}
    values::V
    ε::Float64
    normalized::Bool
end

Base.length(s::SingValSpectrum) = length(s.values)

"""
    schmidt_rank(s::SingValSpectrum) -> Int

Number of kept singular values (= Schmidt rank of the cut).
"""
schmidt_rank(s::SingValSpectrum) = length(s)

"""
    spectral_gap(s::SingValSpectrum) -> Float64

Difference between the two largest singular values, `σ₁ - σ₂`. Returns `σ₁` when the Schmidt
rank is 1 (product state - the gap is effectively infinite, bounded here by the largest value).
"""
function spectral_gap(s::SingValSpectrum)
    return length(s.values) >= 2 ? s.values[1] - s.values[2] : s.values[1]
end

"""
    entanglement_entropy(s::SingValSpectrum; base=2) -> Float64

Von Neumann entanglement entropy ``S_b = -\\sum_i \\sigma_i^2 \\log_b \\sigma_i^2``. Probabilities
``p_i = \\sigma_i^2`` are normalized (``\\sum_i p_i = 1``) before the sum, so this is correct even
for a truncated or non-canonical spectrum whose values don't already sum to unit norm. The
convention ``0 \\log_b 0 := 0`` is enforced so zero singular values never produce `NaN`.
"""
function entanglement_entropy(s::SingValSpectrum; base=2)
    p = abs2.(s.values)
    p ./= sum(p)
    return -sum(pᵢ -> pᵢ > 0 ? pᵢ * log(base, pᵢ) : 0.0, p)
end

"""
    entanglement_spectrum(s::SingValSpectrum) -> Vector{Float64}

The entanglement spectrum ``\\{\\varepsilon_i\\} = -2 \\ln \\sigma_i`` (natural log). Reveals the
level structure of the reduced density matrix - more information than the scalar entropy alone.
"""
entanglement_spectrum(s::SingValSpectrum) = -2.0 .* log.(s.values)

"""
    local_truncation_error(s::SingValSpectrum) -> Float64
    local_truncation_error(ε::Real) -> Float64

The Frobenius-norm-based local truncation error at this bond: by Eckart-Young, the 2-norm of the
discarded singular values IS the Frobenius-norm distance between the truncated and untruncated
factorization at this bond. The `SingValSpectrum` method just reads `s.ε`; the bare-`Real` method
is the identity - it exists so error tracking can be written uniformly against "whatever ε I
have," whether or not a full `SingValSpectrum` was ever built for this bond (a
`SimStudy.QuadratureTruncationErrorAccumulator` never constructs one, reading `ε` straight off
`factorize`'s own return).
"""
local_truncation_error(s::SingValSpectrum) = s.ε
local_truncation_error(ε::Real) = ε

"""
    global_truncation_error(local_errors; nrm=1.0) -> Float64

Combine per-bond [`local_truncation_error`](@ref) values into one sweep-wide truncation error via
quadrature (`hypot`) - errors are 2-norms, so ``\\varepsilon^2`` (discarded weight) adds across
bonds, not ``\\varepsilon`` itself. Divide by `nrm` (the sweep's overall tracked norm) to express
the result as a *relative* tolerance (global rtol) rather than an absolute one.
"""
global_truncation_error(local_errors; nrm::Real=1.0) = hypot(local_errors...) / nrm
