@testset "TIx: tensor index" begin
    @testset "local dimensions" begin
        @testset "dim" begin
            # for a typical spin-1/2 site
            @test dim(TIx{Upper}(:σ_up, 2)) == 2
            @test dim(TIx{Lower}(:σ_down, 2)) == 2

            # index of a leg at the first/last site with a single value
            @test dim(TIx{Upper}(:σ_end, 1)) == 1
        end
        @testset "dim is a positive integer d>0" begin
            @test_throws ArgumentError TIx{Upper}(:α, 0)
            @test_throws ArgumentError TIx{Upper}(:α, -1)
            # @TODO: dim must be an integer
        end
    end

    @testset "label" begin
        @test label(TIx{Upper}(:α_up, 2)) == :α_up
        @test label(TIx{Lower}(:α_down, 2)) == :α_down
        @test label(TIx{Upper}(:α, 2)) isa Symbol
    end

    @testset "index variance is semantic" begin
        α_1 = TIx{Upper}(:α, 4)
        α_2 = TIx{Upper}(:α, 4)
        α_3 = TIx{Lower}(:α, 4)

        @test α_1 == α_2
        @test α_1 != α_3
    end
end

@testset "Quick constructor functions" begin
    @testset "uppers/lowers: no arguments yields empty tuple" begin
        # important for uses like uppers(filter(...)...) where the filter might return nothing
        @test uppers() === ()
        @test lowers() === ()
    end

    @testset "uppers/lowers: duplicate labels are allowed" begin
        a, b = uppers(:α => 2, :α => 3)
        @test a.label == :α && dim(a) == 2
        @test b.label == :α && dim(b) == 3
    end

    @testset "dim is a positive integer d>0" begin
        @test_throws ArgumentError upper(:σ, 0)
        @test_throws ArgumentError lower(:β, -3)
        # @TODO: dim must be an integer
    end

    @testset "uppers_range/lowers_range: default start=1" begin
        # uppers_range with default start
        indices = uppers_range(:α, 2, 3)
        @test length(indices) == 3
        @test indices[1].label == Symbol(:α, :_, 1)
        @test indices[2].label == Symbol(:α, :_, 2)
        @test indices[3].label == Symbol(:α, :_, 3)
        @test all(dim.(indices) .== 2)

        # lowers_range with default start
        indices = lowers_range(:β, 3, 2)
        @test length(indices) == 2
        @test indices[1].label == Symbol(:β, :_, 1)
        @test indices[2].label == Symbol(:β, :_, 2)
        @test all(dim.(indices) .== 3)
    end

    @testset "uppers_range/lowers_range: custom start" begin
        # uppers_range with custom start
        indices = uppers_range(:γ, 4, 5, 2)
        @test length(indices) == 4
        @test indices[1].label == Symbol(:γ, :_, 2)
        @test indices[2].label == Symbol(:γ, :_, 3)
        @test indices[3].label == Symbol(:γ, :_, 4)
        @test indices[4].label == Symbol(:γ, :_, 5)
        @test all(dim.(indices) .== 4)

        # lowers_range with custom start
        indices = lowers_range(:δ, 2, 7, 4)
        @test length(indices) == 4
        @test indices[1].label == Symbol(:δ, :_, 4)
        @test indices[2].label == Symbol(:δ, :_, 5)
        @test indices[3].label == Symbol(:δ, :_, 6)
        @test indices[4].label == Symbol(:δ, :_, 7)
        @test all(dim.(indices) .== 2)
    end

    @testset "uppers_range/lowers_range: variance is correct" begin
        upper_indices = uppers_range(:α, 2, 3)
        lower_indices = lowers_range(:β, 2, 3)

        @test all(typeof(ix) <: TIx{Upper} for ix in upper_indices)
        @test all(typeof(ix) <: TIx{Lower} for ix in lower_indices)
    end
end

@testset "Index notation convention" begin
    @test which_space(TIx{Upper}(:α, 2)) == :domain
    @test which_space(TIx{Lower}(:α, 2)) == :codomain

    # variance is part of identity — same label/dim, different location ≠ same index
    @test TIx{Upper}(:α, 2) != TIx{Lower}(:α, 2)
end
