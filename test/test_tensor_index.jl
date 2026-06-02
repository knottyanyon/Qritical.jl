@testset "TIx" begin

    @testset "ndim" begin
        @test ndim(TIx{Upper}(:σ, 2)) == 2
        @test ndim(TIx{Lower}(:α, 3)) == 3
        @test ndim(TIx{Upper}(:vL, 1)) == 1   # boundary bond
    end

    @testset "direction is semantically meaningful" begin
        @test TIx{Upper}(:α, 4) != TIx{Lower}(:α, 4)
        @test TIx{Upper}(:α, 4) == TIx{Upper}(:α, 4)
    end

end

@testset "Edge cases" begin

    @testset "TIx: ndim must be positive" begin
        @test_throws ArgumentError TIx{Upper}(:α, 0)
        @test_throws ArgumentError TIx{Upper}(:α, -1)
        @test_throws ArgumentError upper(:σ, 0)
        @test_throws ArgumentError lower(:β, -3)
    end

    @testset "MultiIx: empty constituents yields ndim 1" begin
        g = MultiIx(:empty, ())
        @test ndim(g) == 1
    end

    @testset "uppers/lowers: no arguments yields empty tuple" begin
        @test uppers() === ()
        @test lowers() === ()
    end

    @testset "uppers/lowers: duplicate labels are allowed" begin
        a, b = uppers(:α => 2, :α => 3)
        @test a.label == :α && ndim(a) == 2
        @test b.label == :α && ndim(b) == 3
    end

end

@testset "upper / lower constructors" begin

    @testset "single index" begin
        σ = upper(:σ, 2)
        α = lower(:α, 4)
        @test σ == TIx{Upper}(:σ, 2)
        @test α == TIx{Lower}(:α, 4)
        @test ndim(σ) == 2
        @test ndim(α) == 4
    end

    @testset "uppers / lowers batch" begin
        vL, vR = uppers(:vL => 4, :vR => 4)
        @test vL == TIx{Upper}(:vL, 4)
        @test vR == TIx{Upper}(:vR, 4)

        (σ,) = lowers(:σ => 2)
        @test σ == TIx{Lower}(:σ, 2)
    end

end

@testset "MultiIx" begin

    @testset "ndim is the product of constituent ndims" begin
        idx_α = TIx{Upper}(:α, 2)
        idx_β = TIx{Lower}(:β, 3)
        g = MultiIx(:αβ, (idx_α, idx_β))
        @test ndim(g) == ndim(idx_α) * ndim(idx_β)
        @test ndim(g) == 6
    end

    @testset "single constituent: ndim passes through" begin
        idx = TIx{Upper}(:α, 5)
        g = MultiIx(:α, (idx,))
        @test ndim(g) == ndim(idx)
    end

    @testset "triple constituent" begin
        a = TIx{Upper}(:a, 2)
        b = TIx{Lower}(:b, 3)
        c = TIx{Upper}(:c, 4)
        g = MultiIx(:abc, (a, b, c))
        @test ndim(g) == 24
    end

end
