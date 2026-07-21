# ── Partition and Bipartition ─────────────────────────────────────────────────
# A Partition is an ordered list of legs; a Bipartition splits all legs of a
# tensor into two groups (left = rows, right = columns) for an SVD reshape.

@testset "Partition: an ordered list of legs" begin
    @testset "Partition is a plain Vector of AbstractIx" begin
        vL = upper(:vL, 2)
        σ  = lower(:σ, 3)
        p  = Partition([vL, σ])
        @test p isa Vector{AbstractIx}
        @test length(p) == 2
        @test p[1] == vL
        @test p[2] == σ
    end

    @testset "legs can be retrieved by position and compared by equality" begin
        α = upper(:α, 4)
        β = lower(:β, 2)
        p = Partition([α, β])
        @test p[1] == upper(:α, 4)
        @test p[2] == lower(:β, 2)
    end

    @testset "an empty Partition is allowed" begin
        p = Partition([])
        @test p isa Vector{AbstractIx}
        @test isempty(p)
    end
end

@testset "Bipartition: splitting legs into a left group (rows) and a right group (columns)" begin
    @testset "disjoint left and right groups are accepted" begin
        vL = upper(:vL, 2)
        σ  = lower(:σ, 3)
        vR = lower(:vR, 4)
        bp = Bipartition(Partition([vL, σ]), Partition([vR]))
        @test length(bp.left)  == 2
        @test length(bp.right) == 1
        @test bp.left[1]  == vL
        @test bp.right[1] == vR
    end

    @testset "a leg appearing in both groups is rejected" begin
        σ = lower(:σ, 2)
        @test_throws ArgumentError Bipartition(Partition([σ]), Partition([σ]))
    end

    @testset "a leg can appear in both groups when they share a leg with the same label but different variance" begin
        # upper(:σ,2) and lower(:σ,2) are different AbstractIx values — both allowed
        σ_up = upper(:σ, 2)
        σ_lo = lower(:σ, 2)
        @test σ_up != σ_lo
        bp = Bipartition(Partition([σ_up]), Partition([σ_lo]))
        @test length(bp.left) == 1 && length(bp.right) == 1
    end

    @testset "empty left group is allowed" begin
        vL = upper(:vL, 2)
        vR = lower(:vR, 4)
        bp = Bipartition(Partition([]), Partition([vL, vR]))
        @test isempty(bp.left)
        @test length(bp.right) == 2
    end

    @testset "empty right group is allowed" begin
        vL = upper(:vL, 2)
        vR = lower(:vR, 4)
        bp = Bipartition(Partition([vL, vR]), Partition([]))
        @test length(bp.left) == 2
        @test isempty(bp.right)
    end
end

@testset "complement: legs of a tensor not included in a given partition" begin
    @testset "returns the legs not in the partition, in original order" begin
        vL = upper(:vL, 2)
        σ  = lower(:σ, 3)
        vR = lower(:vR, 4)
        c = complement(Partition([vL, σ]), [vL, σ, vR])
        @test length(c) == 1
        @test c[1] == vR
    end

    @testset "empty partition: all legs are returned" begin
        vL = upper(:vL, 2)
        σ  = lower(:σ, 3)
        c = complement(Partition([]), [vL, σ])
        @test c == [vL, σ]
    end

    @testset "full partition: complement is empty" begin
        vL = upper(:vL, 2)
        σ  = lower(:σ, 3)
        c = complement(Partition([vL, σ]), [vL, σ])
        @test isempty(c)
    end

    @testset "original order is preserved in the result" begin
        a = upper(:a, 2)
        b = lower(:b, 3)
        c_ix = upper(:c, 4)
        d = lower(:d, 5)
        # remove b and d; a and c should come back in their original positions
        result = complement(Partition([b, d]), [a, b, c_ix, d])
        @test result == [a, c_ix]
    end
end

@testset "bipartition convenience: right side is automatically the complement" begin
    @testset "right side equals complement of left in the full index list" begin
        vL = upper(:vL, 2)
        σ  = lower(:σ, 3)
        vR = lower(:vR, 4)
        indices = [vL, σ, vR]
        bp = bipartition(Partition([vL, σ]), indices)
        @test bp.left  == [vL, σ]
        @test bp.right == [vR]
    end

    @testset "empty left: right gets all legs" begin
        vL = upper(:vL, 2)
        vR = lower(:vR, 4)
        bp = bipartition(Partition([]), [vL, vR])
        @test isempty(bp.left)
        @test bp.right == [vL, vR]
    end
end
