@testitem "MulTIx: a grouped collection of TIx" begin   # `@testitem "name" begin...end` = Python `class TestMulTIx:` but hierarchical; reports failures per group
    @testitem "MulTIx: empty constituents yields dim 1" begin
        g = MulTIx(:empty, ())   # `MulTIx(label, indices)` = 2-field struct: a Symbol label and a Tuple of AbstractIx; `()` = empty Tuple
        @test dim(g) == 1   # `dim(g) = prod(dim, g.indices; init=1)` = product over empty set = 1 (empty product convention); Python: `math.prod([]) == 1`
    end

    @testitem "Order of indices in MulTIx should matter" begin
        idx_α = TIx(:α, 2)   # Upper index α with dim=2 (spin-1/2 physical leg)
        idx_β = TIx(:β, 3)   # Lower index β with dim=3 (bond leg)
        @test MulTIx(:αβ, (idx_α, idx_β)) != MulTIx(:αβ, (idx_β, idx_α))   # order matters for MulTIx equality: (α,β) ≠ (β,α); Julia's column-major reshape fuses indices in listed order so swapping changes the leg layout; `!=` = not equal 
    end

    @testitem "dim is the product of constituent dims" begin
        @testitem "single constituent: dim passes through" begin
            idx = TIx(:α, 5)   # single index with dim=5; physical: a 5-dimensional site index
            g = MulTIx(:α, (idx,))    # `(idx,)` = 1-tuple ; single-element tuple
            @test dim(g) == dim(idx)  # fusing a single leg: dim(fused) = dim(original) = 5; the product of a single element is that element
        end
        @testitem "two constituents" begin
            idx_α = TIx(:α, 2)   # dim=2
            idx_β = TIx(:β, 3)   # dim=3

            g = MulTIx(:αβ, (idx_α, idx_β))   # fuse α and β into a single combined index; matrix notation: (α,β) fused row index has dim = 2×3 = 6

            @test dim(g) == dim(idx_α) * dim(idx_β)   # dim(fused) = product of constituent dims; `*` = multiplication
            @test dim(g) == 6   # concrete value: 2×3 = 6; used to check group_legs reshape correctness
        end

        @testitem "three constituents" begin
            idx_a = TIx(:a, 2)   # dim=2
            idx_b = TIx(:b, 3)   # dim=3
            idx_c = TIx(:c, 4)   # dim=4
            g = MulTIx(:abc, (idx_a, idx_b, idx_c))   # fuse 3 indices; dim = 2×3×4 = 24; this is what group_legs creates when reshaping a rank-3 partition into a matrix row
            @test dim(g) == 24   # 2×3×4 = 24; fused dimension is the product of all three
        end
    end
end
