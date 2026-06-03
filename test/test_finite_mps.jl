# v0.3 tests: FiniteMPS — physics behavior, algorithms, edge cases

using LinearAlgebra: norm, I

# ── helpers ───────────────────────────────────────────────────────────────────

# Return a random FiniteMPS with uniform bond dimension χ
function _rand_mps(::Type{D}, L::Int, χ::Int; T::Type=Float64) where {D<:AbstractDoF}
    return FiniteMPS(D(), L, χ; T)
end

# ── construction ──────────────────────────────────────────────────────────────

@testset "FiniteMPS — construction" begin
    mps = _rand_mps(Spin{1//2}, 4, 3)
    @test mps.form == ArbitraryForm()
    @test mps.L    == 4
    @test length(mps.tensors)  == 4
    @test length(mps.bond_svs) == 5           # L+1 entries
    @test mps.bond_svs[1]   == [1.0]          # trivial left boundary
    @test mps.bond_svs[end] == [1.0]          # trivial right boundary

    for t in mps.tensors
        @test t isa IndexedTensor{Float64, 3}
    end

    # physical dimension matches hilbert_space
    d = hilbert_space(Spin{1//2}())
    for t in mps.tensors
        @test size(t.data, 2) == d
    end

    # boundary bond dimensions are 1
    @test size(mps.tensors[1].data, 1)   == 1
    @test size(mps.tensors[end].data, 3) == 1
end

@testset "FiniteMPS — complex eltype" begin
    mps = _rand_mps(Fermionic, 3, 2; T=ComplexF64)
    @test mps.bond_svs[1]   == [1.0]
    @test mps.bond_svs[end] == [1.0]
    @test all(t -> eltype(t.data) == ComplexF64, mps.tensors)
end

# ── left canonical sweep ──────────────────────────────────────────────────────

@testset "FiniteMPS — left_canonical_sweep!" begin
    L = 5
    mps = _rand_mps(Spin{1//2}, L, 4)
    left_canonical_sweep!(mps)

    @test mps.form == CanonicalForm(L, L + 1)

    for i in 1:L
        data = mps.tensors[i].data
        χL, d, χR = size(data)
        M = reshape(data, χL * d, χR)
        @test M' * M ≈ I(χR) atol = 1e-12  # left isometry
    end

    @test mps.bond_svs[1]   == [1.0]
    @test mps.bond_svs[end] == [1.0]

    # after normalization ⟨ψ|ψ⟩ ≈ 1
    @test overlap(mps, mps) ≈ 1.0 atol = 1e-12
end

# ── right canonical sweep ─────────────────────────────────────────────────────

@testset "FiniteMPS — right_canonical_sweep!" begin
    L = 5
    mps = _rand_mps(Spin{1//2}, L, 4)
    right_canonical_sweep!(mps)

    @test mps.form == CanonicalForm(0, 1)

    for i in 1:L
        data = mps.tensors[i].data
        χL, d, χR = size(data)
        M = reshape(data, χL, d * χR)
        @test M * M' ≈ I(χL) atol = 1e-12  # right isometry
    end

    @test mps.bond_svs[1]   == [1.0]
    @test mps.bond_svs[end] == [1.0]

    @test overlap(mps, mps) ≈ 1.0 atol = 1e-12
end

# ── both sweeps give the same state ──────────────────────────────────────────

@testset "FiniteMPS — left and right canonical represent same state" begin
    L = 4
    psi_L = _rand_mps(Spin{1//2}, L, 3)
    psi_R = deepcopy(psi_L)

    left_canonical_sweep!(psi_L)
    right_canonical_sweep!(psi_R)

    ovlp = overlap(psi_L, psi_R)
    @test abs(ovlp)^2 ≈ 1.0 atol = 1e-10
end

# ── overlap ───────────────────────────────────────────────────────────────────

@testset "FiniteMPS — overlap" begin
    mps = _rand_mps(Spin{1//2}, 4, 3)
    left_canonical_sweep!(mps)
    @test overlap(mps, mps) ≈ 1.0 atol = 1e-12  # normalized MPS
end

# ── move_center! ──────────────────────────────────────────────────────────────

@testset "FiniteMPS — move_center!" begin
    L = 5
    mps = _rand_mps(Spin{1//2}, L, 3)
    left_canonical_sweep!(mps)
    before = deepcopy(mps)

    # Move center to site 3: sites 1..2 left-canonical, sites 4..5 right-canonical
    move_center!(mps, 3)
    @test mps.form == CanonicalForm(2, 4)
    @test abs(overlap(mps, before))^2 ≈ 1.0 atol = 1e-10

    # From center at 3, move one step right: llim +1, rlim +1
    move_center!(mps, 4)
    @test mps.form == CanonicalForm(3, 5)

    # From center at 4, move one step left: llim -1, rlim -1
    move_center!(mps, 3)
    @test mps.form == CanonicalForm(2, 4)
end

# ── entanglement entropy ──────────────────────────────────────────────────────

@testset "FiniteMPS — entanglement entropy" begin
    # product state: χ=1 everywhere → entropy 0 at all bonds
    L = 4
    mps = _rand_mps(Spin{1//2}, L, 1)
    left_canonical_sweep!(mps)
    for bond in 2:L
        @test entanglement_entropy(mps, bond) ≈ 0.0 atol = 1e-12
    end

    # general state: entropy formula -∑ λᵢ² log(λᵢ²) matches direct computation
    mps2 = _rand_mps(Spin{1//2}, 4, 4)
    left_canonical_sweep!(mps2)
    bond = 3
    svs = mps2.bond_svs[bond]
    nrm = norm(svs)
    λ   = svs ./ nrm
    expected = -sum(x -> x^2 * log(x^2), λ)
    @test entanglement_entropy(mps2, bond) ≈ expected atol = 1e-12
end

# ── MPS addition ──────────────────────────────────────────────────────────────

@testset "FiniteMPS — addition" begin
    L = 3
    ψ = _rand_mps(Spin{1//2}, L, 2)
    φ = _rand_mps(Spin{1//2}, L, 2)
    left_canonical_sweep!(ψ)
    left_canonical_sweep!(φ)

    # a|ψ⟩ + b|φ⟩ — check linearity via overlaps
    a, b = 0.7, 0.3
    χ_sum = a * ψ + b * φ
    @test χ_sum isa FiniteMPS

    # ⟨ψ | aψ + bφ⟩ ≈ a (since ψ is normalized and ⟨ψ|φ⟩ might not be 0)
    @test overlap(ψ, χ_sum) ≈ a * overlap(ψ, ψ) + b * overlap(ψ, φ) atol = 1e-10
end

# ── edge cases ────────────────────────────────────────────────────────────────

@testset "FiniteMPS — L=1 edge case" begin
    mps = _rand_mps(Spin{1//2}, 1, 1)
    @test mps.L == 1
    @test size(mps.tensors[1].data) == (1, 2, 1)
    left_canonical_sweep!(mps)
    @test mps.form == CanonicalForm(1, 2)
    @test overlap(mps, mps) ≈ 1.0 atol = 1e-12
end

@testset "FiniteMPS — CanonicalForm equality" begin
    @test CanonicalForm(3, 5)    == CanonicalForm(3, 5)
    @test CanonicalForm(3, 5)    != CanonicalForm(3, 4)
    @test ArbitraryForm()        == ArbitraryForm()
    @test ArbitraryForm()        != CanonicalForm(0, 1)
end

@testset "FiniteMPS — truncation during sweep" begin
    # KeepFirst(1) on an entangled state → product state, bond svs all [...]
    L = 4
    mps = _rand_mps(Spin{1//2}, L, 4)
    left_canonical_sweep!(mps; trunc=KeepFirst(1))
    @test mps.form == CanonicalForm(L, L + 1)
    # all interior bonds should have χ = 1
    for t in mps.tensors
        @test size(t.data, 3) == 1 || t === mps.tensors[end]
    end
    # entropy should be 0 at all bonds (product state after χ=1 truncation)
    for bond in 2:L
        @test entanglement_entropy(mps, bond) ≈ 0.0 atol = 1e-10
    end
end
