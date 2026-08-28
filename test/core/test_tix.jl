@testitem "TIx: tensor index" begin   # `@testitem "name" begin ... end` = Python `class TestTIx(unittest.TestCase):`; Julia groups tests hierarchically; failures are reported per-testset
    @testitem "local dimensions" begin
        @testitem "dim" begin
            # for a typical spin-1/2 site
            @test dim(TIx(:σ_up, 2)) == 2   # `TIx(:σ_up, 2)` = construct a TIx with label :σ_up, dim 2; spin-1/2 has 2-dimensional local Hilbert space; `dim(...)` = extract stored dimension
            @test dim(TIx(:σ_down, 2)) == 2   # same shape, different label

            # index of a leg at the first/last site with a single value
            @test dim(TIx(:σ_end, 1)) == 1   # dim=1 is the trivial (boundary) case; used for OBC boundary bonds where χ=1
        end
        @testitem "dim is a positive integer d>0" begin
            @test_throws ArgumentError TIx(:α, 0)    # `@test_throws ExceptionType expr` = verify that constructing with dim=0 throws ArgumentError; Julia: raise, Python: `pytest.raises`
            @test_throws ArgumentError TIx(:α, -1)   # negative dim also invalid; inner constructor checks `dim > 0`
            # @TODO: dim must be an integer
        end
    end

    @testitem "label" begin
        @test label(TIx(:α_up, 2)) == :α_up     # `label(ix)` returns a Symbol; `:α_up` is a Symbol literal
        @test label(TIx(:σ, 2)) isa Symbol       # `isa Symbol` = isinstance check ; confirms the return type
    end

    @testitem "equality is label and dim" begin
        α_1 = TIx(:α, 4)
        α_2 = TIx(:α, 4)   # same as α_1 - should compare equal
        α_3 = TIx(:β, 4)   # different label
        α_4 = TIx(:α, 2)   # same label, different dim

        @test α_1 == α_2   # `==` uses `Base.:(==)` which we extended; two TIx with same label/dim are equal
        @test α_1 != α_3   # different label -> never equal
        @test α_1 != α_4   # different dim -> never equal
    end
end

@testitem "Quick constructor functions" begin
    @testitem "ixs: no arguments yields empty tuple" begin
        # important for uses like ixs(filter(...)...) where the filter might return nothing
        @test ixs() === ()   # `===` = identity check (same object); `()` = empty Tuple; `ixs()` with no arguments returns the empty Tuple; Python: `ixs() is ()` (though Python tuple identity varies)
    end

    @testitem "ixs: duplicate labels are allowed" begin
        a, b = ixs(:α => 2, :α => 3)   # `a, b = ixs(...)` = destructure the returned Tuple into two variables
        @test a.label == :α && dim(a) == 2   # `&&` = AND. field access `.label` returns Symbol; check both label and dim
        @test b.label == :α && dim(b) == 3   # same label but different dim -> distinct indices
    end

    @testitem "dim is a positive integer d>0" begin
        @test_throws ArgumentError first(ixs(:σ => 0))    # `ixs` delegates to `TIx` inner constructor which checks dim > 0
        @test_throws ArgumentError first(ixs(:β => -3))   # negative dim also invalid
        # @TODO: dim must be an integer
    end

    @testitem "ixs_range: default start=1" begin
        indices = ixs_range(:α, 2, 3)   # `ixs_range(base, dim, last)` creates indices :α_1, :α_2, :α_3 all with dim=2
        @test length(indices) == 3   # should generate 3 indices (from 1 to 3 inclusive)
        @test indices[1].label == Symbol(:α, :_, 1)   # `Symbol(:α, :_, 1)` = :α_1 (symbol concatenation); `indices[1]` = first element (1-indexed)
        @test indices[2].label == Symbol(:α, :_, 2)   # :α_2
        @test indices[3].label == Symbol(:α, :_, 3)   # :α_3
        @test all(dim.(indices) .== 2)   # `dim.(indices)` = broadcast dim over all indices; `.==` = element-wise equality; `all(...)` = all true

        indices = ixs_range(:β, 3, 2)   # creates indices :β_1, :β_2 both with dim=3
        @test length(indices) == 2
        @test indices[1].label == Symbol(:β, :_, 1)
        @test indices[2].label == Symbol(:β, :_, 2)
        @test all(dim.(indices) .== 3)
    end

    @testitem "ixs_range: custom start" begin
        indices = ixs_range(:γ, 4, 5, 2)   # 4th arg = start=2; creates :γ_2, :γ_3, :γ_4, :γ_5 with dim=4
        @test length(indices) == 4
        @test indices[1].label == Symbol(:γ, :_, 2)   # starts at 2
        @test indices[2].label == Symbol(:γ, :_, 3)
        @test indices[3].label == Symbol(:γ, :_, 4)
        @test indices[4].label == Symbol(:γ, :_, 5)
        @test all(dim.(indices) .== 4)

        indices = ixs_range(:δ, 2, 7, 4)   # start=4, end=7; creates :δ_4, :δ_5, :δ_6, :δ_7 with dim=2
        @test length(indices) == 4
        @test indices[1].label == Symbol(:δ, :_, 4)
        @test indices[2].label == Symbol(:δ, :_, 5)
        @test indices[3].label == Symbol(:δ, :_, 6)
        @test indices[4].label == Symbol(:δ, :_, 7)
        @test all(dim.(indices) .== 2)
    end

    @testitem "ixs_range: element type is TIx" begin
        indices = ixs_range(:α, 2, 3)   # all 3 indices should be TIx
        @test all(typeof(ix) <: TIx for ix in indices)   # `for ix in indices` = generator expression; `<: TIx` = subtype check ; all must be TIx
    end
end
