"""
Tests for: Leg (src/topograph/leg.jl).
"""

@testset "Topograph: Leg owns the variance" begin   # Leg carries the per-end TIx (variance-tagged); the space itself is a cached view of Wire.space, not duplicated data
    @testset "Leg{V} stores variance as a type parameter" begin
        ℓ = Leg(TIx{Upper}(:σ, 2), WireId(1), NodeId(1), 1)   # `Leg(ix, wire, owner, axis)` = positional constructor; `ℓ` (Unicode \ell) avoids shadowing the `l`/`1` visual clash; `TIx{Upper}(:σ, 2)` = an existing elementary index, Upper-variance, dim 2
        @test ℓ isa Leg{Upper}     # the type parameter `V` is inferred from `ix`'s variance. `Leg{Upper}`, not just `Leg` so `V` is resolvable at compile time once `ix`'s concrete type is known
        @test ℓ.ix == TIx{Upper}(:σ, 2)   # `.ix` round-trips the exact TIx object (compared by TIx's own `==`, which checks label+dim+variance)
        @test ℓ.wire == WireId(1)   # `.wire` is the REVERSE key into the wire table not the Wire object itself, just its id
        @test ℓ.owner == NodeId(1)   # `.owner` names which node this leg belongs to. needed for `legs(g, n)` lookups
        @test ℓ.axis == 1           # `.axis` = the array-position this leg occupies on its owning tensor (polarisation, level 2)
    end

    @testset "make_leg builds the right variance from a tensor's own indices" begin
        A = QTensor(randn(3, 4), (upper(:i, 3), lower(:j, 4)))
        ℓ1 = make_leg(A, 1, WireId(1), NodeId(1))
        ℓ2 = make_leg(A, 2, WireId(2), NodeId(1))
        @test ℓ1 isa Leg{Upper}
        @test ℓ2 isa Leg{Lower}
        @test ℓ1.ix == upper(:i, 3)
        @test ℓ2.ix == lower(:j, 4)
        @test ℓ1.axis == 1
        @test ℓ2.axis == 2
    end

    @testset "make_leg contains its instability to exactly one dispatch, at the A.indices[k] boundary" begin
        # A.indices::NTuple{Valence,AbstractIx}, so A.indices[k] is statically only an AbstractIx, this is the first inference boundary named in the topograph design. The outer make_leg is therefore genuinely NOT @inferred-clean, that is expected, not a bug, Base.return_types confirms it can only ever return the abstract Leg. What the function-barrier pattern buys is that _make_leg, once handed a concrete ix, is fully inferred, that is the check that matters and the one @inferred makes.
        A = QTensor(randn(3, 4), (upper(:i, 3), lower(:j, 4)))
        @test only(Base.return_types(make_leg, (typeof(A), Int, WireId, NodeId))) == Leg
        ℓ1 = @inferred Qritical._make_leg(A.indices[1], WireId(1), NodeId(1), 1)   # internal, unexported, so qualified here
        @test ℓ1 isa Leg{Upper}
    end

    @testset "make_leg on a grouped (MulTIx) leg is expected to fail, deferred to M-C" begin
        A = QTensor(randn(3, 4), (upper(:i, 3), lower(:j, 4)))
        bp = Bipartition(Partition([A.indices[1]]), Partition([A.indices[2]]))
        M = group_legs(A, bp)
        @test_throws MethodError make_leg(M, 1, WireId(1), NodeId(1))
    end
end
