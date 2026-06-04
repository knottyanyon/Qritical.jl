# v0.7 tests: Exact Diagonalization — DenseHamiltonian, dense_hamiltonian, ground_state

using LinearAlgebra
using SparseArrays

# ── helpers ───────────────────────────────────────────────────────────────────

# Full L-site Heisenberg Hamiltonian as a dense matrix (big-endian basis).
# Used only as a reference; never builds exponentially large arrays for L > 6.
# Embeds the two-site matrix h2 at each bond (i, i+1) via Kronecker products:
#   I_{2^{i-1}} ⊗ h2 ⊗ I_{2^{L-i-1}}
function _heisenberg_full_matrix(L::Int; J::Float64=1.0)
    d  = 2
    H  = zeros(d^L, d^L)
    h2 = _heisenberg_h2(; J, T=Float64)   # from test_tebd.jl (loaded earlier)
    for i in 1:L-1
        IL = Matrix{Float64}(I, d^(i-1), d^(i-1))
        IR = Matrix{Float64}(I, d^(L-i-1), d^(L-i-1))
        H += kron(IL, kron(h2, IR))
    end
    return H
end

# ── construction ─────────────────────────────────────────────────────────────

@testset "DenseHamiltonian — size(H.H) == (2^L, 2^L)" begin
    for L in [2, 3, 5]
        H = dense_hamiltonian(L, Spin{1//2}())
        @test size(H.H) == (2^L, 2^L)
    end
end

@testset "DenseHamiltonian — H.H is a SparseMatrixCSC" begin
    H = dense_hamiltonian(4, Spin{1//2}())
    @test H.H isa SparseMatrixCSC
end

# ── physics behavior ──────────────────────────────────────────────────────────

@testset "DenseHamiltonian — L=2 matches known 4×4 Heisenberg matrix" begin
    H        = dense_hamiltonian(2, Spin{1//2}(); J=1.0)
    expected = _heisenberg_full_matrix(2; J=1.0)
    @test Matrix(H.H) ≈ expected atol=1e-12
end

@testset "DenseHamiltonian — matrix is symmetric" begin
    for L in [3, 4]
        H = dense_hamiltonian(L, Spin{1//2}())
        @test Matrix(H.H) ≈ Matrix(H.H)' atol=1e-12
    end
end

@testset "DenseHamiltonian — matches dense reference for L=4" begin
    L        = 4
    H        = dense_hamiltonian(L, Spin{1//2}(); J=1.5)
    expected = _heisenberg_full_matrix(L; J=1.5)
    @test Matrix(H.H) ≈ expected atol=1e-12
end

@testset "DenseHamiltonian — nnz ≤ 3L·2^L (sparse scaling)" begin
    for L in [4, 6, 8]
        H = dense_hamiltonian(L, Spin{1//2}())
        @test nnz(H.H) ≤ 3 * L * 2^L
    end
end

@testset "ground_state — Rayleigh quotient equals eigensolver energy" begin
    H = dense_hamiltonian(4, Spin{1//2}())
    E, ψ = ground_state(H)
    @test real(ψ' * H.H * ψ) ≈ E atol=1e-10
end

@testset "ground_state — is real for real Hamiltonian" begin
    H = dense_hamiltonian(4, Spin{1//2}())
    E, ψ = ground_state(H)
    @test E isa Real
    @test eltype(ψ) <: Real
end

@testset "ground_state — ED energy agrees with MPS imaginary time (L=4)" begin
    L        = 4
    H_ed     = dense_hamiltonian(L, Spin{1//2}(); J=1.0)
    E_ed, _  = ground_state(H_ed)

    H_bonds  = _heisenberg_bonds(L)   # from test_tebd.jl
    mps_gs   = time_evolve(FiniteMPS(Spin{1//2}(), L, 16), H_bonds, 5.0, 0.05; imag=true)
    E_mps    = real(expectation_value(mps_gs, heisenberg_mpo(L; J=1.0)))

    @test E_ed ≈ E_mps atol=0.01
end

# ── edge cases ────────────────────────────────────────────────────────────────

@testset "DenseHamiltonian — L=1: zero matrix (no interaction bonds)" begin
    H = dense_hamiltonian(1, Spin{1//2}(); J=1.0)
    @test size(H.H) == (2, 2)
    @test norm(H.H) ≈ 0.0 atol=1e-12
end

@testset "DenseHamiltonian — J=0: zero Hamiltonian, ground state energy = 0" begin
    H = dense_hamiltonian(4, Spin{1//2}(); J=0.0)
    @test norm(H.H) ≈ 0.0 atol=1e-12
    E, _ = ground_state(H)
    @test E ≈ 0.0 atol=1e-10
end

@testset "DenseHamiltonian — L=14 builds and ground state energy is negative" begin
    H = dense_hamiltonian(14, Spin{1//2}(); J=1.0)
    @test size(H.H) == (2^14, 2^14)
    E, ψ = ground_state(H)
    @test E < 0
    @test norm(ψ) ≈ 1.0 atol=1e-10
end
