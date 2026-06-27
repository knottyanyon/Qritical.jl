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

@testset "Index notation convention" begin
    @test which_space(TIx{Upper}(:α, 2)) == :domain
    @test which_space(TIx{Lower}(:α, 2)) == :codomain

    # variance is part of identity — same label/dim, different location ≠ same index
    @test TIx{Upper}(:α, 2) != TIx{Lower}(:α, 2)
end

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

@testset "bond_label: positional label generator" begin
    @test bond_label(:χ, 3) == :χ3
    @test bond_label(:α, 1) == :α1
    @test bond_label(:χ, 12) == :χ12
    @test bond_label(:λ, 0) == :λ0
end
