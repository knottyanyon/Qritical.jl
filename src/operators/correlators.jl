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
function overlap(ψ::FiniteMPS, φ::FiniteMPS)   # compute ⟨ψ|φ⟩; both MPS must have the same chain length L and the same physical dimension d at every site
    length(ψ.tensors) == length(φ.tensors) || throw(   # check L matches; `||` short-circuit: only throws if the condition is false; `throw(ArgumentError(...))` raises a typed exception 
        ArgumentError(
            "overlap: MPS lengths must match, got $(length(ψ.tensors)) and $(length(φ.tensors))",   # `$(...)` is Julia string interpolation 
        ),
    )
    all(   # `all(gen)` returns true if every element of the generator is true 
        size(ψ.tensors[i].data, 2) == size(φ.tensors[i].data, 2) for   # check physical dimension d at every site; `size(A, 2)` = dimension along index 2 
        i in 1:length(ψ.tensors)   # `1:length(...)` is an inclusive range 
    ) || throw(ArgumentError("overlap: physical dimensions must match at every site"))
    L = length(ψ.tensors)   # chain length
    env = ones(promote_type(eltype(ψ.tensors[1].data), eltype(φ.tensors[1].data)), 1, 1)   # initialise 1×1 environment = scalar 1.0; `promote_type(T1, T2)` finds the common type (e.g. ComplexF64 wins over Float64); `ones(T, 1, 1)` = 1×1 matrix of ones; shape grows as we sweep right
    for i in 1:L   # sweep left-to-right site by site
        Aψ = ψ.tensors[i].data    # (χLψ, d, χRψ)  # bra MPS tensor at site i; `.data` extracts the raw 3D array from the QTensor wrapper
        Aφ = φ.tensors[i].data    # (χLφ, d, χRφ)  # ket MPS tensor at site i
        d = size(Aψ, 2)   # physical dimension at this site
        χRψ = size(Aψ, 3)   # right bond dimension of ψ
        χRφ = size(Aφ, 3)   # right bond dimension of φ
        new_env = zeros(eltype(env), χRψ, χRφ)   # new environment has shape (χR_ψ, χR_φ); `zeros(T, m, n)` = Python `np.zeros((m,n), dtype=T)`
        for σ in 1:d   # sum over physical index σ 
            new_env += Aψ[:, σ, :]' * env * Aφ[:, σ, :]   # update env: Aψ†·env·Aφ; `A[:, σ, :]` slices the σ-th physical layer → (χL, χR) matrix; `A'` is the conjugate transpose. matrix product `*` contracts the bond indices
        end
        env = new_env   # advance environment one site to the right
    end
    return env[1, 1]   # after all L sites: scalar result ⟨ψ|φ⟩ (boundary bond dims are 1)
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
function local_expectation(ψ::FiniteMPS, op::AbstractMatrix, site::Int)   # compute ⟨ψ|O_site|ψ⟩; `op` is a d×d matrix; `AbstractMatrix` accepts any matrix type (dense, sparse, etc.)
    L = length(ψ.tensors)   # chain length
    env = ones(promote_type(eltype(ψ.tensors[1].data), eltype(op)), 1, 1)   # initialise 1×1 scalar environment; `promote_type` promotes to a common numeric type (so if op is Float64 and A is ComplexF64, result is ComplexF64)
    for i in 1:L   # sweep through all sites
        A = ψ.tensors[i].data    # (χL, d, χR)  # MPS tensor at site i
        d = size(A, 2)   # physical dimension at this site
        χR = size(A, 3)   # right bond dimension
        new_env = zeros(eltype(env), χR, χR)   # new environment; initialise to zero before accumulating σ contributions
        for σ in 1:d, σ′ in 1:d   # double loop over physical indices σ (bra) and σ′ (ket); `for σ in 1:d, σ′ in 1:d` is a combined nested loop 
            weight =
                (i == site) ? op[σ, σ′] : (σ == σ′ ? one(eltype(op)) : zero(eltype(op)))   # ternary: if at the operator site use op[σ,σ′]; otherwise use δ(σ,σ′) (identity insert); `one(T)` and `zero(T)` are type-generic 1 and 0 
            iszero(weight) && continue   # skip zero-weight terms to avoid useless matrix multiplications; `iszero(x)` is true if x is exactly zero; `&& continue` short-circuits to the next loop iteration 
            new_env += weight * (A[:, σ, :]' * env * A[:, σ′, :])   # accumulate: weight × A†_{σ} · env · A_{σ′}; note σ for bra (conjugated) and σ′ for ket (not conjugated)
        end
        env = new_env   # advance environment
    end
    return env[1, 1]   # scalar ⟨ψ|O_site|ψ⟩
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
function two_site_op(
    ψ::FiniteMPS, op_i::AbstractMatrix, op_j::AbstractMatrix, i::Int, j::Int
)   # compute ⟨ψ|O_i O_j|ψ⟩; physics: this is the two-point correlator C(i,j) = ⟨O_i O_j⟩
    L = length(ψ.tensors)   # chain length
    T = promote_type(eltype(ψ.tensors[1].data), eltype(op_i), eltype(op_j))   # promote to a type that can hold all three; `promote_type(T1,T2,T3)` finds the common supertype 
    env = ones(T, 1, 1)   # initialise 1×1 environment
    for site in 1:L   # sweep through all L sites
        A = ψ.tensors[site].data   # MPS tensor at current site; shape (χL, d, χR)
        d = size(A, 2)   # physical dimension
        χR = size(A, 3)   # right bond dimension
        new_env = zeros(T, χR, χR)   # new environment, initialised to zero
        for σ in 1:d, σ′ in 1:d   # double loop over physical indices
            weight = if site == i   # Julia multi-line if-expression (like a ternary but more readable); returns the weight W[σ,σ′] for this site
                op_i[σ, σ′]   # insert op_i at site i (the left operator of the two-point correlator)
            elseif site == j
                op_j[σ, σ′]   # insert op_j at site j (the right operator)
            else
                σ == σ′ ? one(T) : zero(T)   # everywhere else: insert identity δ(σ,σ′); `one(T)` = T(1), `zero(T)` = T(0)
            end
            iszero(weight) && continue   # skip zero contributions (optimization)
            new_env += weight * (A[:, σ, :]' * env * A[:, σ′, :])   # transfer matrix step: weight × A†_σ · env · A_σ′; repeated for all non-zero (σ,σ′) pairs
        end
        env = new_env   # advance environment one step to the right
    end
    return env[1, 1]   # final scalar = ⟨ψ|O_i O_j|ψ⟩; this is the raw two-point correlator (not connected)
end
