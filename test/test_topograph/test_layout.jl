@testset "§6.1 AbstractLayout, Chain — layout specifications" begin

    @testset "Chain <: AbstractLayout" begin
        @test Chain <: AbstractLayout
    end

    @testset "Chain constructors" begin
        g = Chain(6)
        @test g.L == 6
        @test g.periodic == false

        g_pbc = Chain(6, true)
        @test g_pbc.periodic == true
    end

    @testset "sites" begin
        g = Chain(5)
        @test sites(g) == 1:5
        @test length(sites(g)) == 5
    end

    @testset "bonds — open boundary (default)" begin
        g = Chain(4)
        bs = bonds(g)
        # OBC: L-1 = 3 nearest-neighbour pairs
        @test length(bs) == 3
        @test bs == [(1,2), (2,3), (3,4)]
    end

    @testset "bonds — periodic boundary" begin
        g = Chain(4, true)
        bs = bonds(g)
        # PBC: L = 4 bonds, including (4,1) wrap
        @test length(bs) == 4
        @test (4, 1) ∈ bs || (1, 4) ∈ bs   # wrap bond present
    end

    @testset "Chain(1) edge case" begin
        g = Chain(1)
        @test length(sites(g)) == 1
        @test length(bonds(g)) == 0          # no bonds for a single site
    end

    @testset "bonds are consistent with sites" begin
        g = Chain(6)
        all_sites = Set(sites(g))
        for (i, j) in bonds(g)
            @test i ∈ all_sites
            @test j ∈ all_sites
            @test i != j
        end
    end

end
