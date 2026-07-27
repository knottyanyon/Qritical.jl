@testset "TIx: tensor index" begin   # `@testset "name" begin ... end` = Python `class TestTIx(unittest.TestCase):`; Julia groups tests hierarchically; failures are reported per-testset
    @testset "local dimensions" begin
        @testset "dim" begin
            # for a typical spin-1/2 site
            @test dim(TIx{Upper}(:σ_up, 2)) == 2   # `TIx{Upper}(:σ_up, 2)` = construct a TIx with Upper variance, label :σ_up, dim 2; spin-1/2 has 2-dimensional local Hilbert space; `dim(...)` = extract stored dimension
            @test dim(TIx{Lower}(:σ_down, 2)) == 2   # same for Lower variance; both Upper and Lower have dim = 2 (variance only affects contraction role, not size)

            # index of a leg at the first/last site with a single value
            @test dim(TIx{Upper}(:σ_end, 1)) == 1   # dim=1 is the trivial (boundary) case; used for OBC boundary bonds where χ=1
        end
        @testset "dim is a positive integer d>0" begin
            @test_throws ArgumentError TIx{Upper}(:α, 0)    # `@test_throws ExceptionType expr` = verify that constructing with dim=0 throws ArgumentError; Julia: raise, Python: `pytest.raises`
            @test_throws ArgumentError TIx{Upper}(:α, -1)   # negative dim also invalid; inner constructor checks `dim > 0`
            # @TODO: dim must be an integer
        end
    end

    @testset "label" begin
        @test label(TIx{Upper}(:α_up, 2)) == :α_up     # `label(ix)` returns a Symbol; `:α_up` is a Symbol literal
        @test label(TIx{Lower}(:α_up, 2)) == :α_up     # label is independent of variance — same label with different variance
        @test label(TIx{Upper}(:α, 2)) isa Symbol       # `isa Symbol` = isinstance check ; confirms the return type
    end

    @testset "index variance is semantic" begin
        α_1 = TIx{Upper}(:α, 4)
        α_2 = TIx{Upper}(:α, 4)   # same as α_1 — should compare equal
        α_3 = TIx{Lower}(:α, 4)   # same label/dim but different variance (Lower)

        @test α_1 == α_2   # `==` uses `Base.:(==)` which we extended; two TIx{Upper} with same label/dim are equal
        @test α_1 != α_3   # `!=` = Python `!=`; different variance → never equal (TIx equality requires same type parameter L)
    end
end

@testset "Quick constructor functions" begin
    @testset "uppers/lowers: no arguments yields empty tuple" begin
        # important for uses like uppers(filter(...)...) where the filter might return nothing
        @test uppers() === ()   # `===` = identity check (same object); `()` = empty Tuple; `uppers()` with no arguments returns the empty Tuple; Python: `uppers() is ()` (though Python tuple identity varies)
        @test lowers() === ()   # same for lowers()
    end

    @testset "uppers/lowers: duplicate labels are allowed" begin
        a, b = uppers(:α => 2, :α => 3)   # `a, b = uppers(...)` = destructure the returned Tuple into two variables 
        @test a.label == :α && dim(a) == 2   # `&&` = AND. field access `.label` returns Symbol; check both label and dim
        @test b.label == :α && dim(b) == 3   # same label but different dim → distinct indices
    end

    @testset "dim is a positive integer d>0" begin
        @test_throws ArgumentError upper(:σ, 0)    # `upper` delegates to `TIx{Upper}` inner constructor which checks dim > 0
        @test_throws ArgumentError lower(:β, -3)   # negative dim also invalid
        # @TODO: dim must be an integer
    end

    @testset "uppers_range/lowers_range: default start=1" begin
        # uppers_range with default start
        indices = uppers_range(:α, 2, 3)   # `uppers_range(base, dim, last)` creates Upper indices :α_1, :α_2, :α_3 all with dim=2
        @test length(indices) == 3   # should generate 3 indices (from 1 to 3 inclusive)
        @test indices[1].label == Symbol(:α, :_, 1)   # `Symbol(:α, :_, 1)` = :α_1 (symbol concatenation); `indices[1]` = first element (1-indexed)
        @test indices[2].label == Symbol(:α, :_, 2)   # :α_2
        @test indices[3].label == Symbol(:α, :_, 3)   # :α_3
        @test all(dim.(indices) .== 2)   # `dim.(indices)` = broadcast dim over all indices; `.==` = element-wise equality; `all(...)` = all true 

        # lowers_range with default start
        indices = lowers_range(:β, 3, 2)   # creates Lower indices :β_1, :β_2 both with dim=3
        @test length(indices) == 2
        @test indices[1].label == Symbol(:β, :_, 1)
        @test indices[2].label == Symbol(:β, :_, 2)
        @test all(dim.(indices) .== 3)
    end

    @testset "uppers_range/lowers_range: custom start" begin
        # uppers_range with custom start
        indices = uppers_range(:γ, 4, 5, 2)   # 4th arg = start=2; creates :γ_2, :γ_3, :γ_4, :γ_5 with dim=4
        @test length(indices) == 4
        @test indices[1].label == Symbol(:γ, :_, 2)   # starts at 2
        @test indices[2].label == Symbol(:γ, :_, 3)
        @test indices[3].label == Symbol(:γ, :_, 4)
        @test indices[4].label == Symbol(:γ, :_, 5)
        @test all(dim.(indices) .== 4)

        # lowers_range with custom start
        indices = lowers_range(:δ, 2, 7, 4)   # start=4, end=7; creates :δ_4, :δ_5, :δ_6, :δ_7 with dim=2
        @test length(indices) == 4
        @test indices[1].label == Symbol(:δ, :_, 4)
        @test indices[2].label == Symbol(:δ, :_, 5)
        @test indices[3].label == Symbol(:δ, :_, 6)
        @test indices[4].label == Symbol(:δ, :_, 7)
        @test all(dim.(indices) .== 2)
    end

    @testset "uppers_range/lowers_range: variance is correct" begin
        upper_indices = uppers_range(:α, 2, 3)   # all 3 indices should be TIx{Upper}
        lower_indices = lowers_range(:β, 2, 3)   # all 3 should be TIx{Lower}

        @test all(typeof(ix) <: TIx{Upper} for ix in upper_indices)   # `for ix in upper_indices` = generator expression; `<: TIx{Upper}` = subtype check ; all must be Upper
        @test all(typeof(ix) <: TIx{Lower} for ix in lower_indices)   # all must be Lower
    end
end

@testset "Index notation convention" begin
    @test which_space(TIx{Upper}(:α, 2)) == :domain    # Upper = incoming arrow = domain = contravariant coefficient index; physics: ket indices A^σ are Upper
    @test which_space(TIx{Lower}(:α, 2)) == :codomain  # Lower = outgoing arrow = codomain = covariant basis index; physics: bra indices A_σ are Lower

    # variance is part of identity — same label/dim, different location ≠ same index
    @test TIx{Upper}(:α, 2) != TIx{Lower}(:α, 2)   # equality requires SAME label, dim, AND variance; this is the key design choice: variance is NOT just a label — it's part of the index's mathematical identity
end
