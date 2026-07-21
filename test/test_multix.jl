@testset "MulTIx: a grouped collection of TIx" begin
    @testset "MulTIx: empty constituents yields dim 1" begin
        g = MulTIx(:empty, ())
        @test dim(g) == 1
    end

    @testset "Order of indices in MulTIx should matter" begin
        idx_α = TIx{Upper}(:α, 2)
        idx_β = TIx{Lower}(:β, 3)
        @test MulTIx(:αβ, (idx_α, idx_β)) != MulTIx(:αβ, (idx_β, idx_α))
    end

    @testset "dim is the product of constituent dims" begin
        @testset "single constituent: dim passes through" begin
            idx = TIx{Upper}(:α, 5)
            g = MulTIx(:α, (idx,))
            @test dim(g) == dim(idx)
        end
        @testset "two constituents" begin
            idx_α = TIx{Upper}(:α, 2)
            idx_β = TIx{Lower}(:β, 3)

            g = MulTIx(:αβ, (idx_α, idx_β))

            @test dim(g) == dim(idx_α) * dim(idx_β)
            @test dim(g) == 6
        end

        @testset "three constituents" begin
            idx_a = TIx{Upper}(:a, 2)
            idx_b = TIx{Lower}(:b, 3)
            idx_c = TIx{Upper}(:c, 4)
            g = MulTIx(:abc, (idx_a, idx_b, idx_c))
            @test dim(g) == 24
        end
    end
end
