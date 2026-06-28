@testset "§6.1 DoF layer — types, operators, statistics" begin

    @testset "local_dim" begin
        @test local_dim(Spin{1//2}()) == 2
        @test local_dim(Spin{1}())    == 3
        @test local_dim(SpinlessFermion()) == 2
        @test local_dim(Electron())        == 4
        @test local_dim(HardCoreBoson())   == 2
        @test local_dim(Majorana())        == 2
    end

    @testset "statistics" begin
        # spins and hard-core bosons: commuting inter-site algebra
        @test canonical_relation(Spin{1//2}())   isa CCR
        @test canonical_relation(Spin{1}())      isa CCR
        @test canonical_relation(HardCoreBoson()) isa CCR
        # fermions and Majorana: anticommuting inter-site algebra
        @test canonical_relation(SpinlessFermion()) isa CAR
        @test canonical_relation(Electron())        isa CAR
        @test canonical_relation(Majorana())        isa CAR
    end

    @testset "physical_space — sectorless (NoSymmetry)" begin
        # For now: returns a plain integer dimension (sectorless ComplexSpace)
        @test physical_space(Spin{1//2}(),    NoSymmetry()) == 2
        @test physical_space(Spin{1}(),       NoSymmetry()) == 3
        @test physical_space(SpinlessFermion(), NoSymmetry()) == 2
        @test physical_space(Electron(),        NoSymmetry()) == 4
        @test physical_space(HardCoreBoson(),   NoSymmetry()) == 2
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Spin-½ operator physics
    # ─────────────────────────────────────────────────────────────────────────
    @testset "algebra_generators(SpinHalf()) — NamedTuple keys" begin
        ops = algebra_generators(Spin{1//2}())
        @test haskey(ops, :Sx)
        @test haskey(ops, :Sy)
        @test haskey(ops, :Sz)
        @test haskey(ops, :Sp)
        @test haskey(ops, :Sm)
        @test haskey(ops, :I)
    end

    @testset "algebra_generators(SpinHalf()) — matrix sizes are 2×2" begin
        ops = algebra_generators(Spin{1//2}())
        for name in (:Sx, :Sy, :Sz, :Sp, :Sm, :I)
            M = getproperty(ops, name)
            @test size(M) == (2, 2)
        end
    end

    @testset "algebra_generators(SpinHalf()) — Sᶻ eigenvalues are ±½" begin
        ops = algebra_generators(Spin{1//2}())
        ev = sort(real.(eigvals(ops.Sz)))
        @test ev ≈ [-0.5, 0.5] atol=1e-12
    end

    @testset "algebra_generators(SpinHalf()) — SU(2) commutation relations" begin
        ops = algebra_generators(Spin{1//2}())
        Sx, Sy, Sz = ops.Sx, ops.Sy, ops.Sz
        # [Sx, Sy] = i Sz
        @test Sx*Sy - Sy*Sx ≈ im * Sz  atol=1e-12
        # [Sy, Sz] = i Sx
        @test Sy*Sz - Sz*Sy ≈ im * Sx  atol=1e-12
        # [Sz, Sx] = i Sy
        @test Sz*Sx - Sx*Sz ≈ im * Sy  atol=1e-12
    end

    @testset "algebra_generators(SpinHalf()) — ladder operator adjoint (S⁺)† = S⁻" begin
        ops = algebra_generators(Spin{1//2}())
        @test ops.Sp' ≈ ops.Sm  atol=1e-12
        @test ops.Sm' ≈ ops.Sp  atol=1e-12
    end

    @testset "algebra_generators(SpinHalf()) — ladder operators raise/lower Sᶻ" begin
        ops = algebra_generators(Spin{1//2}())
        # S⁺ raises: S⁺ |↓⟩ = |↑⟩   (column 2 of S⁺ is the up-spin basis vector)
        down = [0.0; 1.0]   # |↓⟩ in {|↑⟩,|↓⟩} basis
        up   = [1.0; 0.0]   # |↑⟩
        @test ops.Sp * down ≈ up   atol=1e-12
        @test ops.Sm * up   ≈ down atol=1e-12
        # annihilate at boundary
        @test norm(ops.Sp * up)   < 1e-12
        @test norm(ops.Sm * down) < 1e-12
    end

    @testset "algebra_generators(SpinHalf()) — Casimir S² = s(s+1)I = 3/4 I" begin
        ops = algebra_generators(Spin{1//2}())
        S2 = ops.Sx^2 + ops.Sy^2 + ops.Sz^2
        @test S2 ≈ (3/4) * ops.I  atol=1e-12
    end

    @testset "algebra_generators(SpinHalf()) — Sx, Sz are Hermitian; Sy is Hermitian" begin
        ops = algebra_generators(Spin{1//2}())
        @test ops.Sx ≈ ops.Sx'  atol=1e-12
        @test ops.Sy ≈ ops.Sy'  atol=1e-12
        @test ops.Sz ≈ ops.Sz'  atol=1e-12
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Spin-1 operator physics
    # ─────────────────────────────────────────────────────────────────────────
    @testset "algebra_generators(Spin{1}()) — 3×3 matrices with Casimir s(s+1)=2" begin
        ops = algebra_generators(Spin{1}())
        for name in (:Sx, :Sy, :Sz, :Sp, :Sm, :I)
            @test size(getproperty(ops, name)) == (3, 3)
        end
        S2 = ops.Sx^2 + ops.Sy^2 + ops.Sz^2
        @test S2 ≈ 2 * ops.I  atol=1e-12   # s(s+1) = 1·2 = 2
        ev = sort(real.(eigvals(ops.Sz)))
        @test ev ≈ [-1.0, 0.0, 1.0]  atol=1e-12
    end

    # ─────────────────────────────────────────────────────────────────────────
    # SpinlessFermion operator physics
    # ─────────────────────────────────────────────────────────────────────────
    @testset "algebra_generators(SpinlessFermion()) — CAR: {c, c†} = I, {c,c} = 0" begin
        ops = algebra_generators(SpinlessFermion())
        # canonical anticommutation: {c, c†} = cc† + c†c = I
        @test ops.c * ops.cdag + ops.cdag * ops.c ≈ ops.I  atol=1e-12
        # {c, c} = 0
        @test ops.c * ops.c + ops.c * ops.c ≈ zeros(2, 2)  atol=1e-12
        # number operator: n = c†c
        @test ops.cdag * ops.c ≈ ops.n  atol=1e-12
    end

    @testset "algebra_generators(SpinlessFermion()) — n eigenvalues are 0 and 1" begin
        ops = algebra_generators(SpinlessFermion())
        ev = sort(real.(eigvals(ops.n)))
        @test ev ≈ [0.0, 1.0]  atol=1e-12
    end

    # ─────────────────────────────────────────────────────────────────────────
    # HardCoreBoson operator physics
    # ─────────────────────────────────────────────────────────────────────────
    @testset "algebra_generators(HardCoreBoson()) — b² = 0, {b,b†} = I" begin
        ops = algebra_generators(HardCoreBoson())
        @test ops.b * ops.b ≈ zeros(2, 2)  atol=1e-12   # hard-core: no double occupancy
        @test ops.b * ops.bdag + ops.bdag * ops.b ≈ ops.I  atol=1e-12
        @test ops.bdag * ops.b ≈ ops.n  atol=1e-12
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Electron operator physics
    # ─────────────────────────────────────────────────────────────────────────
    @testset "algebra_generators(Electron()) — 4×4 sizes" begin
        ops = algebra_generators(Electron())
        for name in (:cup, :cdn, :cupdag, :cdndag, :nup, :ndn, :n, :I)
            @test size(getproperty(ops, name)) == (4, 4)
        end
    end

    @testset "algebra_generators(Electron()) — CAR within each spin species" begin
        ops = algebra_generators(Electron())
        # {c↑, c†↑} = I, {c↓, c†↓} = I
        @test ops.cup * ops.cupdag + ops.cupdag * ops.cup ≈ ops.I  atol=1e-12
        @test ops.cdn * ops.cdndag + ops.cdndag * ops.cdn ≈ ops.I  atol=1e-12
    end

    @testset "algebra_generators(Electron()) — nup, ndn, n eigenvalues" begin
        ops = algebra_generators(Electron())
        # nup has eigenvalues 0,0,1,1 (two states with up, two without)
        ev_up = sort(real.(eigvals(ops.nup)))
        @test ev_up ≈ [0.0, 0.0, 1.0, 1.0]  atol=1e-12
        ev_n  = sort(real.(eigvals(ops.n)))
        @test ev_n  ≈ [0.0, 1.0, 1.0, 2.0]  atol=1e-12
    end

    @testset "algebra_generators(Electron()) — standard intra-site sign: c↓|↑↓⟩ = −|↑⟩" begin
        ops = algebra_generators(Electron())
        # basis: {|0⟩=e1, |↑⟩=e2, |↓⟩=e3, |↑↓⟩=e4}
        ket_updown = [0.0, 0.0, 0.0, 1.0]   # |↑↓⟩
        ket_up     = [0.0, 1.0, 0.0, 0.0]   # |↑⟩
        @test ops.cdn * ket_updown ≈ -ket_up  atol=1e-12   # the fixed intra-site sign
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Majorana operators physics
    # ─────────────────────────────────────────────────────────────────────────
    @testset "algebra_generators(Majorana()) — {γₐ, γᵦ} = 2δ" begin
        ops = algebra_generators(Majorana())
        # {γ1, γ1} = 2I
        @test ops.γ1 * ops.γ1 + ops.γ1 * ops.γ1 ≈ 2 * ops.I  atol=1e-12
        # {γ2, γ2} = 2I
        @test ops.γ2 * ops.γ2 + ops.γ2 * ops.γ2 ≈ 2 * ops.I  atol=1e-12
        # {γ1, γ2} = 0  (anticommute for different Majoranas)
        @test ops.γ1 * ops.γ2 + ops.γ2 * ops.γ1 ≈ zeros(2,2)  atol=1e-12
    end

    @testset "algebra_generators(Majorana()) — both are Hermitian (γ†=γ)" begin
        ops = algebra_generators(Majorana())
        @test ops.γ1 ≈ ops.γ1'  atol=1e-12
        @test ops.γ2 ≈ ops.γ2'  atol=1e-12
    end

end
