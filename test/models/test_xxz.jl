# Dense reconstruction of an MPState into a linearized d^L state vector, for cross-checking
# neel_state/domain_wall_state against their expected single-nonzero-entry structure.
function _dense_state_vector_neel(mps)
    L = length(mps.sites)
    A = [convert(Array, tensor(mps.sites[i])) for i in 1:L]
    d = size(A[1], 2)
    ψ = zeros(ComplexF64, d^L)
    for idx in Iterators.product(ntuple(_ -> 1:d, L)...)
        vec = [1.0 + 0.0im]
        for i in 1:L
            a = A[i]
            newvec = zeros(ComplexF64, size(a, 3))
            for bin in eachindex(vec), bout in axes(a, 3)
                newvec[bout] += vec[bin] * a[bin, idx[i], bout, 1]
            end
            vec = newvec
        end
        lin = sum((idx[i] - 1) * 2^(i - 1) for i in 1:L) + 1
        ψ[lin] = vec[1]
    end
    return ψ
end

@testitem "xxz_hamiltonian: dense reconstruction matches a hand-built dense XXZ matrix" begin
    using TensorKit

    L = 2
    V = ComplexSpace(2)
    H = xxz_hamiltonian(L, V; Jxy=0.7, Jz=1.3, h=0.0)
    mpo = to_mpo(H)

    A1 = convert(Array, tensor(mpo.sites[1]))
    A2 = convert(Array, tensor(mpo.sites[2]))
    D = size(A1, 3)
    Hmat = zeros(ComplexF64, 4, 4)
    for s1k in 1:2, s2k in 1:2, s1b in 1:2, s2b in 1:2
        ki = s1k + (s2k - 1) * 2
        qi = s1b + (s2b - 1) * 2
        Hmat[ki, qi] = sum(A1[1, s1k, b, s1b] * A2[b, s2k, 1, s2b] for b in 1:D)
    end

    Xm = ComplexF64[0 1; 1 0]
    Ym = ComplexF64[0 -im; im 0]
    Zm = ComplexF64[1 0; 0 -1]
    expected = zeros(ComplexF64, 4, 4)
    for s1k in 1:2, s2k in 1:2, s1b in 1:2, s2b in 1:2
        ki = s1k + (s2k - 1) * 2
        qi = s1b + (s2b - 1) * 2
        expected[ki, qi] =
            0.7 * Xm[s1k, s1b] * Xm[s2k, s2b] +
            0.7 * Ym[s1k, s1b] * Ym[s2k, s2b] +
            1.3 * Zm[s1k, s1b] * Zm[s2k, s2b]
    end
    @test Hmat ≈ expected
end

@testitem "xxz_hamiltonian: Jxy=0 recovers a classical Ising chain (diagonal in the Z basis)" begin
    using TensorKit

    L = 3
    V = ComplexSpace(2)
    H = xxz_hamiltonian(L, V; Jxy=0.0, Jz=1.0, h=0.0)
    mpo = to_mpo(H)
    A1 = convert(Array, tensor(mpo.sites[1]))
    A2 = convert(Array, tensor(mpo.sites[2]))
    A3 = convert(Array, tensor(mpo.sites[3]))
    D1 = size(A1, 3)
    D2 = size(A2, 3)
    # with no XY coupling, H is diagonal in the Z (computational) basis - off-diagonal entries
    # of the dense reconstruction must vanish exactly.
    for s1k in 1:2, s2k in 1:2, s3k in 1:2, s1b in 1:2, s2b in 1:2, s3b in 1:2
        ki = s1k + (s2k - 1) * 2 + (s3k - 1) * 4
        qi = s1b + (s2b - 1) * 2 + (s3b - 1) * 4
        ki == qi && continue
        entry = sum(
            A1[1, s1k, b1, s1b] * A2[b1, s2k, b2, s2b] * A3[b2, s3k, 1, s3b] for
            b1 in 1:D1, b2 in 1:D2
        )
        @test entry ≈ 0.0 atol = 1e-12
    end
end

@testitem "xxz_hamiltonian: h=0 omits field terms entirely" begin
    using TensorKit

    L = 3
    V = ComplexSpace(2)
    H_nofield = xxz_hamiltonian(L, V; h=0.0)
    H_field = xxz_hamiltonian(L, V; h=0.5)
    @test length(H_nofield.terms) == 3 * (L - 1)
    @test length(H_field.terms) == 3 * (L - 1) + L
end

@testitem "neel_state/domain_wall_state: correct alternating/split product structure" begin
    using TensorKit

    L = 4
    V = ComplexSpace(2)

    neel = neel_state(L, V)
    @test neel isa MPState{LeftCanonical,FiniteSupport}
    idx = ntuple(i -> isodd(i) ? 1 : 2, L)
    reconstructed = _dense_state_vector_neel(neel)
    lin = sum((idx[i] - 1) * 2^(i - 1) for i in 1:L) + 1
    @test isapprox(abs(reconstructed[lin]), 1.0; atol=1e-8)

    wall = domain_wall_state(L, V)
    @test wall isa MPState{LeftCanonical,FiniteSupport}
    idx2 = ntuple(i -> i <= L ÷ 2 ? 1 : 2, L)
    reconstructed2 = _dense_state_vector_neel(wall)
    lin2 = sum((idx2[i] - 1) * 2^(i - 1) for i in 1:L) + 1
    @test isapprox(abs(reconstructed2[lin2]), 1.0; atol=1e-8)
end
