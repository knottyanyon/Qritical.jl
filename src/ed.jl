# §10  Exact diagonalisation — dense and sparse solve paths.
#
# `ExactDiagonalization(:ground)` uses KrylovKit Lanczos on the sparse matrix;
# `ExactDiagonalization(:full)` calls dense `eigen(Hermitian(Matrix(H)))`.
# Both paths share `sparse(H)` as the entry point for matrix construction.

import KrylovKit

# ─────────────────────────────────────────────────────────────────────────────
# Sparse matrix from Operator
# ─────────────────────────────────────────────────────────────────────────────

const _ED_MAX_DIM = 2^20   # refuse Hilbert spaces larger than ~1 M

"""
    sparse(H::Operator) -> SparseMatrixCSC{ComplexF64}

Build the sparse ``d^L \\times d^L`` Hamiltonian matrix for the operator `H`.

The matrix is assembled by sparse Kronecker products of local operators onto the
full Hilbert space ``\\mathcal{H} = \\bigotimes_{i=1}^{L} \\mathcal{h}_i``.

Raises `ArgumentError` if ``d^L > 2^{20} \\approx 10^6`` (refuse to silently allocate
multi-GB arrays).
"""
function sparse(H::Operator)
    d = local_dim(H.dof)
    L = H.geom.L
    D = d^L
    D ≤ _ED_MAX_DIM || throw(ArgumentError(
        "Hilbert space dimension $D = $(d)^$L exceeds the safety limit $_ED_MAX_DIM. " *
        "Use an MPS algorithm for large systems."))

    Id = sparse(Matrix{ComplexF64}(I, d, d))
    Sp = SparseArrays.spzeros(ComplexF64, D, D)

    for lt in H.onsite
        mats = [i == lt.site ? sparse(ComplexF64.(lt.op)) : Id for i in 1:L]
        Sp .+= lt.coupling .* foldl(kron, mats)
    end

    for bt in H.bond
        i, j = bt.i, bt.j
        mats = Vector{SparseArrays.SparseMatrixCSC{ComplexF64, Int}}(undef, L)
        for k in 1:L
            if k == i
                mats[k] = sparse(ComplexF64.(bt.op_i))
            elseif k == j
                mats[k] = sparse(ComplexF64.(bt.op_j))
            else
                mats[k] = Id
            end
        end
        Sp .+= bt.coupling .* foldl(kron, mats)
    end

    return Sp
end

# ─────────────────────────────────────────────────────────────────────────────
# Study and algorithm types
# ─────────────────────────────────────────────────────────────────────────────

"""
    GroundState

Study type indicating that the solve target is the lowest eigenvalue (ground state).
Passed as the second argument to `solve`.
"""
struct GroundState end

"""
    ExactDiagonalization{mode}

Algorithm type for exact diagonalisation.

- `ExactDiagonalization(:ground)` — Lanczos on the sparse matrix; returns only
  the ground state energy and state vector. Cost: ``O(d^L \\chi_{\\text{Krylov}})``.
- `ExactDiagonalization(:full)` — full diagonalisation via `eigen(Hermitian(...))`;
  returns all eigenvalues. Cost: ``O(d^{3L})``.
"""
struct ExactDiagonalization{mode}
    ExactDiagonalization(mode::Symbol) = new{mode}()
end

# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

"""
    EDResult

Return type of `solve` with `ExactDiagonalization`.

Fields:
- `energy::Float64`           — ground-state energy (lowest eigenvalue).
- `state::Vector{ComplexF64}` — ground-state vector (unit-norm; empty for `:full`).
- `spectrum::Vector{Float64}` — all eigenvalues (sorted); non-empty only for `:full`.
"""
struct EDResult
    energy::Float64
    state::Vector{ComplexF64}
    spectrum::Vector{Float64}
end

# ─────────────────────────────────────────────────────────────────────────────
# solve dispatch
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve(H, ::GroundState, ::ExactDiagonalization{:ground}) -> EDResult

Compute the ground state of `H` using the Lanczos algorithm on the sparse matrix.

Uses KrylovKit's `eigsolve` with a random initial vector; returns the lowest
eigenvalue and the corresponding eigenvector normalised to unit norm.
"""
function solve(H::Operator, ::GroundState, ::ExactDiagonalization{:ground})
    M   = sparse(H)
    D   = size(M, 1)
    v₀  = normalize(randn(ComplexF64, D))
    # eigsolve returns eigenvalues closest to 0 by default; use `which=:SR` for smallest real
    vals, vecs, _ = KrylovKit.eigsolve(M, v₀, 1, :SR; ishermitian=true, tol=1e-12, maxiter=300)
    E   = real(vals[1])
    ψ   = normalize(vecs[1])
    EDResult(E, ψ, Float64[E])
end

"""
    solve(H, ::GroundState, ::ExactDiagonalization{:full}) -> EDResult

Full diagonalisation of `H`.

Returns all eigenvalues (sorted ascending) in `result.spectrum`.  `result.state`
is the ground-state eigenvector and `result.energy` is the smallest eigenvalue.
"""
function solve(H::Operator, ::GroundState, ::ExactDiagonalization{:full})
    M   = dense_matrix(H)
    F   = eigen(Hermitian(M))
    evs = real.(F.values)
    ψ   = normalize(F.vectors[:, 1])
    EDResult(evs[1], ψ, sort(evs))
end
