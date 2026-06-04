using SparseArrays
using KrylovKit

# ── DenseHamiltonian ──────────────────────────────────────────────────────────

"""
    DenseHamiltonian{D <: AbstractDoF, T <: Number}

Full many-body Hamiltonian stored as a `SparseMatrixCSC{T,Int}` of size
``2^L \\times 2^L`` (for spin-``1/2``). The basis is big-endian: site 1 is
the most significant index, site ``L`` the least significant.

Construct with [`dense_hamiltonian`](@ref); solve with [`ground_state`](@ref).
"""
struct DenseHamiltonian{D <: AbstractDoF, T <: Number}
    H::SparseMatrixCSC{T, Int}
    dof::D
    L::Int
end

# ── two-site operator builder ─────────────────────────────────────────────────

function _spin_h2(::Spin{S}; J::T) where {S, T <: Real}
    d  = Int(2S + 1)
    ms = collect(S:-1:-S)   # [S, S-1, ..., -S]

    Sz = Diagonal(T.(ms))

    Sp = zeros(T, d, d)
    for k in 1:d-1
        m_from = ms[k + 1]       # lower m_s value (state being raised)
        m_to   = ms[k]           # higher m_s value (result)
        Sp[k, k + 1] = sqrt(T((S - m_from) * (S + m_to)))
    end
    Sm = collect(Sp')

    return J * (kron(Matrix(Sz), Matrix(Sz)) + (kron(Sp, Sm) + kron(Sm, Sp)) / 2)
end

# ── dense_hamiltonian ─────────────────────────────────────────────────────────

"""
    dense_hamiltonian(L, dof; J=1.0, T=Float64) -> DenseHamiltonian

Build the Heisenberg XXX Hamiltonian

```math
H = J \\sum_{i=1}^{L-1} \\bigl(S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z\\bigr)
```

as a ``d^L \\times d^L`` sparse matrix (``d = 2S+1`` for `Spin{S}`). Each
two-site term is embedded via

```math
I_{d^{i-1}} \\otimes h_{i,i+1} \\otimes I_{d^{L-i-1}}
```

using `SparseArrays.kron` so sparsity is preserved at every step.
"""
function dense_hamiltonian(
    L::Int, dof::Spin{S};
    J::Real=1.0, T::Type{<:Real}=Float64
) where {S}
    d   = hilbert_space(dof)
    N   = d^L
    H   = spzeros(T, N, N)
    h2  = sparse(_spin_h2(dof; J=T(J)))

    for i in 1:L-1
        IL = sparse(I, d^(i-1),     d^(i-1))
        IR = sparse(I, d^(L-i-1),   d^(L-i-1))
        H += kron(kron(IL, h2), IR)
    end

    return DenseHamiltonian(dropzeros!(H), dof, L)
end

# ── ground_state ──────────────────────────────────────────────────────────────

"""
    ground_state(H::DenseHamiltonian; tol=1e-10) -> (E, ψ)

Find the ground state energy ``E`` and normalized state vector ``\\psi`` using
`KrylovKit.eigsolve` (Lanczos algorithm, `which = :SR`).

Returns real values for a real-symmetric Hamiltonian.
"""
function ground_state(H::DenseHamiltonian{D, T}; tol::Real=1e-10) where {D, T}
    N = size(H.H, 1)

    if iszero(H.H)
        x = normalize!(ones(T, N))
        return zero(T), x
    end

    x0         = randn(T, N)
    vals, vecs, _ = eigsolve(H.H, x0, 1, :SR; tol=tol, ishermitian=true)
    return real(vals[1]), real.(vecs[1])
end
