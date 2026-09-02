@testitem "TIx: tensor index" begin
    import TensorKit: TensorKit, GradedSpace, Z2Irrep   # narrow import: TensorKit also exports
    # `dim`/`space`, which would collide with Qritical's own bindings under a blanket
    # `using TensorKit` in this scope; `TensorKit: TensorKit, ...` still brings in the module
    # name itself for qualified access (`TensorKit.ComplexSpace`, `TensorKit.fuse`, ...)

    @testitem "local dimensions" begin
        @testitem "dim" begin
            # for a typical spin-1/2 site
            @test dim(TIx(2)) == 2                       # trivial (ComplexSpace) leg, dim=2
            @test dim(TIx(1)) == 1                        # dim=1: trivial boundary case, χ=1

            g = TIx(GradedSpace(Z2Irrep(0) => 2, Z2Irrep(1) => 3))
            @test dim(g) == 5                              # symmetric leg: total dim sums across sectors
        end
        @testitem "dim is a positive integer d>0" begin
            @test_throws ArgumentError TIx(0)              # trivial-sector convenience constructor still validates d > 0
            @test_throws ArgumentError TIx(-1)
        end
    end

    @testitem "space" begin
        @test space(TIx(4)) == TensorKit.ComplexSpace(4)   # `space(ix)` returns the wrapped TensorKit.ElementarySpace directly
        @test space(TIx(4)) isa TensorKit.ElementarySpace
    end

    @testitem "label is deliberately unimplemented for TIx" begin
        # TensorKit itself has no concept of a named leg, so TIx matches that model: calling
        # label on a TIx is an intentional error (via AbstractIx's interface stub), not an
        # oversight - it documents that a TIx has no name.
        @test_throws ErrorException label(TIx(4))
    end

    @testitem "equality is space equality" begin
        α_1 = TIx(4)
        α_2 = TIx(4)   # same underlying space as α_1 - should compare equal
        α_3 = TIx(2)   # different dimension -> different space

        @test α_1 == α_2
        @test α_1 != α_3

        g_1 = TIx(GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1))
        g_2 = TIx(GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1))
        g_3 = TIx(GradedSpace(Z2Irrep(0) => 2, Z2Irrep(1) => 1))
        @test g_1 == g_2   # same sector structure
        @test g_1 != g_3   # different multiplicities -> different space
        @test g_1 != α_1   # a graded space is never equal to a trivial ComplexSpace of the same total dim
    end
end

@testitem "MulTIx: a grouped collection of TIx" begin
    import TensorKit: TensorKit, GradedSpace, Z2Irrep   # narrow import: TensorKit also exports
    # `dim`/`space`, which would collide with Qritical's own bindings under a blanket
    # `using TensorKit` in this scope; `TensorKit: TensorKit, ...` still brings in the module
    # name itself for qualified access (`TensorKit.ComplexSpace`, `TensorKit.fuse`, ...)

    @testitem "MulTIx: empty constituents yields the trivial 1-dim space" begin
        g = MulTIx(:empty, ())
        @test dim(g) == 1                              # TensorKit.fuse has no zero-arg method; the trivial space is the sensible default
        @test space(g) == TensorKit.ComplexSpace(1)
    end

    @testitem "Order of indices in MulTIx should matter" begin
        idx_α = TIx(2)
        idx_β = TIx(3)
        @test MulTIx(:αβ, (idx_α, idx_β)) != MulTIx(:αβ, (idx_β, idx_α))   # order matters: swapping changes the fused leg's layout
    end

    @testitem "dim is TensorKit.dim of the fused space" begin
        @testitem "single constituent: dim passes through" begin
            idx = TIx(5)
            g = MulTIx(:α, (idx,))
            @test dim(g) == dim(idx)   # fusing a single leg is the identity on dim
        end
        @testitem "two trivial constituents: dim is the familiar product" begin
            idx_α = TIx(2)
            idx_β = TIx(3)
            g = MulTIx(:αβ, (idx_α, idx_β))
            @test dim(g) == dim(idx_α) * dim(idx_β)   # trivial-sector fusion reduces to the dimension product
            @test dim(g) == 6
        end
        @testitem "three trivial constituents" begin
            idx_a = TIx(2)
            idx_b = TIx(3)
            idx_c = TIx(4)
            g = MulTIx(:abc, (idx_a, idx_b, idx_c))
            @test dim(g) == 24
        end
    end

    @testitem "space is real TensorKit fusion, not a flat dimension product" begin
        # Z2 fusion (charge conservation): (0,0)/(1,1) -> sector 0, (0,1)/(1,0) -> sector 1.
        # A naive dimension product would give one undifferentiated 4-dim block; real fusion
        # splits it into two 2-dim sectors.
        a = TIx(GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1))
        b = TIx(GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1))
        g = MulTIx(:ab, (a, b))
        expected = TensorKit.fuse(space(a), space(b))
        @test space(g) == expected
        @test space(g) == GradedSpace(Z2Irrep(0) => 2, Z2Irrep(1) => 2)
        @test dim(g) == 4
    end

    @testitem "label" begin
        g = MulTIx(:ασ, (TIx(3), TIx(2)))
        @test label(g) == :ασ
        @test label(g) isa Symbol
    end
end
