# v0.4 tests: FiniteMPO — physics behavior, algorithms, edge cases

using LinearAlgebra: norm, I

# ── helpers ───────────────────────────────────────────────────────────────────

# Product state |↑↓↑↓...⟩ as a χ=1 FiniteMPS.
# Basis: index 1 = |↑⟩, index 2 = |↓⟩.
function _neel_mps(L::Int; T::Type=Float64)
    d   = hilbert_space(Spin{1//2}())
    RT  = real(T)
    bond_svs = [RT[1] for _ in 1:L+1]
    tensors  = Vector{IndexedTensor{T, 3}}(undef, L)
    for i in 1:L
        data = zeros(T, 1, d, 1)
        data[1, isodd(i) ? 1 : 2, 1] = one(T)
        vL = upper(bond_label(:χ, i - 1), 1)
        σ  = lower(:σ, d)
        vR = lower(bond_label(:χ, i), 1)
        tensors[i] = IndexedTensor(data, (vL, σ, vR))
    end
    return FiniteMPS{Spin{1//2}, T, RT}(L, tensors, bond_svs, CanonicalForm(L, L + 1))
end

# ── construction ──────────────────────────────────────────────────────────────

@testset "FiniteMPO — construction" begin
    L   = 4
    mpo = heisenberg_mpo(L; J=1.0)

    @test mpo       isa FiniteMPO
    @test mpo.L     == L
    @test length(mpo.tensors) == L

    for t in mpo.tensors
        @test ndims(t.data) == 4          # (vL, σ_ket, σ_bra, vR)
        @test size(t.data, 2) == 2        # d = hilbert_space(Spin{1//2})
        @test size(t.data, 3) == 2
    end

    # Heisenberg MPO virtual bond dimension is 5
    @test size(mpo.tensors[2].data, 1) == 5   # interior left bond
    @test size(mpo.tensors[2].data, 4) == 5   # interior right bond

    # Boundary sites are dimension-1 on the open side
    @test size(mpo.tensors[1].data, 1) == 1   # left boundary
    @test size(mpo.tensors[L].data, 4) == 1   # right boundary
end

# ── physics behavior ──────────────────────────────────────────────────────────

@testset "FiniteMPO — ⟨↑↓|H|↑↓⟩ = -J/4" begin
    # For the 2-site Heisenberg chain with J=1 the Néel state |↑↓⟩ satisfies:
    #   ⟨↑↓|H|↑↓⟩ = J·(S^z_1 S^z_2)|↑↓⟩ = J·(+1/2)(−1/2) = −J/4
    # The S^± flip terms vanish because S^−_2|↓⟩ = 0.
    mps = _neel_mps(2)
    mpo = heisenberg_mpo(2; J=1.0)

    @test expectation_value(mps, mpo) ≈ -1/4 atol=1e-12
end

@testset "FiniteMPO — ⟨ψ|I|ψ⟩ = ⟨ψ|ψ⟩" begin
    L   = 4
    mps = FiniteMPS(Spin{1//2}(), L, 4)
    left_canonical_sweep!(mps)
    mpo = identity_mpo(Spin{1//2}(), L)

    @test expectation_value(mps, mpo) ≈ real(overlap(mps, mps)) atol=1e-12
end

@testset "FiniteMPO — MPO×MPS bond dimension" begin
    L   = 4
    χ   = 3
    mps = FiniteMPS(Spin{1//2}(), L, χ)
    mpo = heisenberg_mpo(L; J=1.0)

    result = apply(mpo, mps)
    @test result isa FiniteMPS

    # Interior bond dim of result = χ_MPS_actual × χ_MPO
    for i in 2:L-1
        χ_mps = size(mps.tensors[i].data, 1)   # actual MPS left-bond dim
        @test size(result.tensors[i].data, 1) == χ_mps * 5
    end
end

@testset "FiniteMPO — variance ⟨H²⟩ - ⟨H⟩² ≥ 0" begin
    L   = 4
    mps = FiniteMPS(Spin{1//2}(), L, 8)
    left_canonical_sweep!(mps)
    mpo = heisenberg_mpo(L; J=1.0)

    E  = expectation_value(mps, mpo)
    H_psi = apply(mpo, mps)
    E2 = real(overlap(H_psi, H_psi))   # ‖H|ψ⟩‖² = ⟨ψ|H²|ψ⟩

    @test E2 - E^2 >= -1e-10           # variance ≥ 0
end

@testset "FiniteMPO — J=0 gives zero energy" begin
    # J=0 Heisenberg is a pure field term; with h=0 the whole Hamiltonian is 0
    L   = 4
    mps = FiniteMPS(Spin{1//2}(), L, 4)
    left_canonical_sweep!(mps)
    mpo = heisenberg_mpo(L; J=0.0)

    @test expectation_value(mps, mpo) ≈ 0.0 atol=1e-12
end

# ── edge cases ────────────────────────────────────────────────────────────────

@testset "FiniteMPO — L=2 has no interior tensors" begin
    mpo = heisenberg_mpo(2; J=1.0)
    @test mpo.L == 2
    @test length(mpo.tensors) == 2
    # only boundary tensors, both dimension 1 on their open side
    @test size(mpo.tensors[1].data, 1) == 1
    @test size(mpo.tensors[2].data, 4) == 1
end

@testset "FiniteMPO — expectation scales with ‖ψ‖²" begin
    # For an unnormalized MPS α|ψ⟩, ⟨αψ|H|αψ⟩ = α² ⟨ψ|H|ψ⟩
    L   = 4
    mps = FiniteMPS(Spin{1//2}(), L, 4)
    mpo = heisenberg_mpo(L; J=1.0)
    α   = 2.5

    E1  = expectation_value(mps, mpo)
    E2  = expectation_value(α * mps, mpo)

    @test E2 ≈ α^2 * E1 atol=1e-10
end
