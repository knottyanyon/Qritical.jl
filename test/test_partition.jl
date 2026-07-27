# ── Partition and Bipartition ─────────────────────────────────────────────────
# A Partition is an ordered list of legs; a Bipartition splits all legs of a
# tensor into two groups (left = rows, right = columns) for an SVD reshape.

@testset "Partition: an ordered list of legs" begin
    @testset "Partition is a plain Vector of AbstractIx" begin
        vL = upper(:vL, 2)   # `upper(:vL, 2)` = TIx{Upper}(:vL, 2); left virtual bond leg; Upper = arrow points into OC
        σ = lower(:σ, 3)    # `lower(:σ, 3)` = TIx{Lower}(:σ, 3); physical site leg; Lower = codomain/outgoing
        p = Partition([vL, σ])   # `Partition([...])` = just `Vector{AbstractIx}([...])` because `Partition = Vector{AbstractIx}`; square brackets = array literal 
        @test p isa Vector{AbstractIx}   # `isa` = isinstance check; confirms Partition is just a plain Vector; there is NO newtype wrapping
        @test length(p) == 2             # `length(p)` = Python `len(p)`; 2 legs in partition
        @test p[1] == vL                 # 1-indexed array access; first element is vL 
        @test p[2] == σ                  # second element is σ    # Upper index with dim=4
        β = lower(:β, 2)   # Lower index with dim=2
        p = Partition([α, β])
        @test p[1] == upper(:α, 4)   # construct a fresh TIx{Upper}(:α,4) and compare; equality checks label + dim + variance
        @test p[2] == lower(:β, 2)   # same for β; confirms equality works across distinct objects
    end

    @testset "an empty Partition is allowed" begin
        p = Partition([])           # `Partition([])` = empty Vector{AbstractIx}; needed for scalar tensors or trivial bipartitions
        @test p isa Vector{AbstractIx}   # still a Vector{AbstractIx} even when empty
        @test isempty(p)            # `isempty(p)` = Python `len(p) == 0`; must be empty
    end
end

@testset "Bipartition: splitting legs into a left group (rows) and a right group (columns)" begin
    @testset "disjoint left and right groups are accepted" begin
        vL = upper(:vL, 2)   # will go into left partition (rows)
        σ = lower(:σ, 3)    # also in left
        vR = lower(:vR, 4)   # in right partition (cols)
        bp = Bipartition(Partition([vL, σ]), Partition([vR]))   # `Bipartition(left, right)` = struct with two Partitions; inner constructor checks disjointness
        @test length(bp.left) == 2   # left has 2 legs: vL and σ
        @test length(bp.right) == 1   # right has 1 leg: vR
        @test bp.left[1] == vL       # preserve order: vL is first
        @test bp.right[1] == vR       # vR is the only right leg
    end

    @testset "a leg appearing in both groups is rejected" begin
        σ = lower(:σ, 2)
        @test_throws ArgumentError Bipartition(Partition([σ]), Partition([σ]))   # `@test_throws ExceptionType expr` = Python: `with pytest.raises(ArgumentError):`; same σ in both left and right must raise ArgumentError because it violates disjointness
    end

    @testset "a leg can appear in both groups when they share a leg with the same label but different variance" begin
        # upper(:σ,2) and lower(:σ,2) are different AbstractIx values — both allowed
        σ_up = upper(:σ, 2)   # TIx{Upper}(:σ, 2) — domain leg
        σ_lo = lower(:σ, 2)   # TIx{Lower}(:σ, 2) — codomain leg; different TYPE PARAMETER L → different object
        @test σ_up != σ_lo   # different variance makes them unequal despite same label/dim; this is the key: TIx equality requires same L
        bp = Bipartition(Partition([σ_up]), Partition([σ_lo]))   # disjointness check compares by TIx equality (which includes variance), so σ_up ≠ σ_lo → accepted
        @test length(bp.left) == 1 && length(bp.right) == 1   # `&&` = AND; both groups have 1 leg each
    end

    @testset "empty left group is allowed" begin
        vL = upper(:vL, 2)
        vR = lower(:vR, 4)
        bp = Bipartition(Partition([]), Partition([vL, vR]))   # left is empty → all legs are "columns"; produces a 1-row matrix when group_legs is called
        @test isempty(bp.left)       # left group is empty
        @test length(bp.right) == 2  # right has both legs
    end

    @testset "empty right group is allowed" begin
        vL = upper(:vL, 2)
        vR = lower(:vR, 4)
        bp = Bipartition(Partition([vL, vR]), Partition([]))   # right is empty → all legs are "rows"; produces a 1-column matrix
        @test length(bp.left) == 2  # left has both legs
        @test isempty(bp.right)     # right group is empty
    end
end

@testset "complement: legs of a tensor not included in a given partition" begin
    @testset "returns the legs not in the partition, in original order" begin
        vL = upper(:vL, 2)
        σ = lower(:σ, 3)
        vR = lower(:vR, 4)
        c = complement(Partition([vL, σ]), [vL, σ, vR])   # `complement(p, indices)` = filter out legs that are in p; returns those NOT in p; input is any Vector{AbstractIx}
        @test length(c) == 1   # only vR is not in [vL, σ]
        @test c[1] == vR       # first (and only) element of complement is vR
    end

    @testset "empty partition: all legs are returned" begin
        vL = upper(:vL, 2)
        σ = lower(:σ, 3)
        c = complement(Partition([]), [vL, σ])   # empty partition → nothing is excluded → all legs returned
        @test c == [vL, σ]   # `==` on arrays compares element-wise; c should contain both vL and σ in order
    end

    @testset "full partition: complement is empty" begin
        vL = upper(:vL, 2)
        σ = lower(:σ, 3)
        c = complement(Partition([vL, σ]), [vL, σ])   # partition contains all legs → nothing left over
        @test isempty(c)   # complement is empty; `isempty` = Python `len(c) == 0`
    end

    @testset "original order is preserved in the result" begin
        a = upper(:a, 2)
        b = lower(:b, 3)
        c_ix = upper(:c, 4)
        d = lower(:d, 5)
        # remove b and d; a and c should come back in their original positions
        result = complement(Partition([b, d]), [a, b, c_ix, d])   # partition={b,d} → complement={a, c}; ORDER must match [a, b, c, d] with b,d removed → [a, c]
        @test result == [a, c_ix]   # a is at position 1, c_ix at position 3 in original; order preserved; Python: `assert result == [a, c_ix]`
    end
end

@testset "bipartition convenience: right side is automatically the complement" begin
    @testset "right side equals complement of left in the full index list" begin
        vL = upper(:vL, 2)
        σ = lower(:σ, 3)
        vR = lower(:vR, 4)
        indices = [vL, σ, vR]
        bp = bipartition(Partition([vL, σ]), indices)   # `bipartition(left, indices)` = convenience: builds Bipartition(left, complement(left, indices)); avoids having to compute complement manually
        @test bp.left == [vL, σ]   # left is exactly what we passed
        @test bp.right == [vR]      # right is automatically computed as complement([vL,σ] in [vL,σ,vR]) = [vR]
    end

    @testset "empty left: right gets all legs" begin
        vL = upper(:vL, 2)
        vR = lower(:vR, 4)
        bp = bipartition(Partition([]), [vL, vR])   # empty left → right = complement([], [vL,vR]) = [vL, vR]
        @test isempty(bp.left)         # left is empty
        @test bp.right == [vL, vR]    # right contains all legs
    end
end
