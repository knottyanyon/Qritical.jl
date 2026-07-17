"""
    overlap(ψ::FiniteMPS, φ::FiniteMPS) -> Number

Compute ``\\langle \\psi | \\varphi \\rangle`` by contracting the bra-ket network
left-to-right using a running environment tensor.

```math
\\langle \\psi | \\varphi \\rangle
= \\sum_{\\boldsymbol{\\sigma}} \\overline{A^{\\psi}_{\\sigma_1} \\cdots A^{\\psi}_{\\sigma_L}}
  \\cdot A^{\\varphi}_{\\sigma_1} \\cdots A^{\\varphi}_{\\sigma_L}
```
"""
function overlap(ψ::FiniteMPS, φ::FiniteMPS)
    length(ψ.tensors) == length(φ.tensors) || throw(
        ArgumentError(
            "overlap: MPS lengths must match, got $(length(ψ.tensors)) and $(length(φ.tensors))",
        ),
    )
    all(
        size(ψ.tensors[i].data, 2) == size(φ.tensors[i].data, 2) for
        i in 1:length(ψ.tensors)
    ) || throw(ArgumentError("overlap: physical dimensions must match at every site"))
    L = length(ψ.tensors)
    env = ones(promote_type(eltype(ψ.tensors[1].data), eltype(φ.tensors[1].data)), 1, 1)
    for i in 1:L
        Aψ = ψ.tensors[i].data    # (χLψ, d, χRψ)
        Aφ = φ.tensors[i].data    # (χLφ, d, χRφ)
        d = size(Aψ, 2)
        χRψ = size(Aψ, 3)
        χRφ = size(Aφ, 3)
        new_env = zeros(eltype(env), χRψ, χRφ)
        for σ in 1:d
            new_env += Aψ[:, σ, :]' * env * Aφ[:, σ, :]
        end
        env = new_env
    end
    return env[1, 1]
end

"""
    local_expectation(ψ::FiniteMPS, op::AbstractMatrix, site::Int) -> Number

Compute the single-site expectation value ``\\langle \\psi | O_\\text{site} | \\psi \\rangle``
by left-to-right environment contraction.

# Algorithm

Build a running left environment ``E_i`` (shape ``\\chi \\times \\chi``) site by site:

```math
E_i[\\alpha, \\beta] = \\sum_{\\alpha', \\beta', \\sigma, \\sigma'}
    \\overline{A_i[\\alpha', \\sigma, \\alpha]} \\; O_{\\sigma \\sigma'} \\; A_i[\\alpha', \\sigma', \\beta] \\; E_{i-1}[\\alpha', \\beta']
```

At sites other than `site`, insert the identity instead of `op` (which is the
standard transfer-matrix step, collapsing to ``I`` for canonical tensors).
After the last site, the ``1 \\times 1`` environment is the expectation value.

# Why this is O(Lχ²d)

The per-site cost is one ``(\\chi d \\times \\chi)`` matrix multiply — ``O(\\chi^2 d)`` —
regardless of system size.  For a left-canonical MPS the left environment is
exactly ``I`` up to the operator insertion site, so only the right half needs
to be contracted explicitly.  The full left-to-right pass is used here for
correctness on arbitrary canonical forms.
"""
function local_expectation(ψ::FiniteMPS, op::AbstractMatrix, site::Int)
    L = length(ψ.tensors)
    env = ones(promote_type(eltype(ψ.tensors[1].data), eltype(op)), 1, 1)
    for i in 1:L
        A = ψ.tensors[i].data    # (χL, d, χR)
        d = size(A, 2)
        χR = size(A, 3)
        new_env = zeros(eltype(env), χR, χR)
        for σ in 1:d, σ′ in 1:d
            weight =
                (i == site) ? op[σ, σ′] : (σ == σ′ ? one(eltype(op)) : zero(eltype(op)))
            iszero(weight) && continue
            new_env += weight * (A[:, σ, :]' * env * A[:, σ′, :])
        end
        env = new_env
    end
    return env[1, 1]
end

"""
    two_site_op(ψ::FiniteMPS, op_i::AbstractMatrix, op_j::AbstractMatrix, i::Int, j::Int) -> Number

Compute the two-site expectation value ``\\langle \\psi | O_i O_j | \\psi \\rangle``
by a single left-to-right environment pass with two operator insertions.

At each site the running environment is updated by the transfer matrix:

```math
E_{\\text{site}} = \\sum_{\\sigma, \\sigma'} W_{\\sigma \\sigma'} \\; A^\\dagger_{\\sigma} \\, E_{\\text{prev}} \\, A_{\\sigma'}
```

where ``W = \\texttt{op\\_i}`` at site ``i``, ``W = \\texttt{op\\_j}`` at site ``j``,
and ``W = I`` everywhere else.  The cost is ``O(L \\chi^2 d)`` — identical to
[`local_expectation`](@ref) because the second operator insertion adds no extra
contraction steps.

For the connected correlator ``\\langle O_i O_j \\rangle_c = \\langle O_i O_j \\rangle - \\langle O_i \\rangle \\langle O_j \\rangle``,
compute `two_site_op` and subtract the product of two `local_expectation` calls.
"""
function two_site_op(ψ::FiniteMPS, op_i::AbstractMatrix, op_j::AbstractMatrix, i::Int, j::Int)
    L = length(ψ.tensors)
    T = promote_type(eltype(ψ.tensors[1].data), eltype(op_i), eltype(op_j))
    env = ones(T, 1, 1)
    for site in 1:L
        A = ψ.tensors[site].data
        d = size(A, 2)
        χR = size(A, 3)
        new_env = zeros(T, χR, χR)
        for σ in 1:d, σ′ in 1:d
            weight = if site == i
                op_i[σ, σ′]
            elseif site == j
                op_j[σ, σ′]
            else
                σ == σ′ ? one(T) : zero(T)
            end
            iszero(weight) && continue
            new_env += weight * (A[:, σ, :]' * env * A[:, σ′, :])
        end
        env = new_env
    end
    return env[1, 1]
end
