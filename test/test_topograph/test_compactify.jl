"""
Tests for: compactify, boundary, is_ordinary, is_closed (src/topograph/compactify.jl).
"""

@testset "Topograph: compactify and boundary" begin
    @testset "a pinned wire is untouched by compactify" begin
        g, w, ℓA, ℓB = mk_pair(lower(:vL, 4), upper(:vL, 4))
        pin!(g, ℓA, ℓB)
        before = length(nodes(g))
        ĝ = compactify(g)
        @test length(nodes(ĝ)) == before   # no outer node needed, both ends already attached
        @test attachment(ĝ.wires[w]) === Pinned()
        @test boundary(g) == NodeId[]
    end

    @testset "a half-loose wire gains exactly one outer node, and becomes pinned" begin
        g = TensorNetwork()
        n = add_node!(g)
        w = add_wire!(g, 2; label=:σ)
        A = QTensor(randn(2, 2), (upper(:σ, 2), lower(:τ, 2)))
        ℓ = add_leg!(g, A, n, w)
        attach!(g, ℓ)   # add_leg! alone only registers the leg, attach! is what actually sets the wire's end
        @test attachment(g.wires[w]) === HalfLoose()

        ĝ = compactify(g)
        @test length(nodes(ĝ)) == length(nodes(g)) + 1
        @test attachment(ĝ.wires[w]) === Pinned()
        @test boundary(g) == setdiff(nodes(ĝ), nodes(g))
        @test length(boundary(g)) == 1
    end

    @testset "a loose wire gains exactly two outer nodes, and becomes pinned" begin
        g = TensorNetwork()
        w = add_wire!(g, 3; label=:free)   # never attached to any tensor
        @test attachment(g.wires[w]) === Loose()

        ĝ = compactify(g)
        @test length(nodes(ĝ)) == 2
        @test attachment(ĝ.wires[w]) === Pinned()
        @test length(boundary(g)) == 2
    end

    @testset "a circle is untouched by compactify, it stays a Circle" begin
        g = TensorNetwork()
        w = add_wire!(g, 5; label=:loop, closed=true)
        ĝ = compactify(g)
        @test length(nodes(ĝ)) == length(nodes(g))
        @test attachment(ĝ.wires[w]) === Circle()
        @test boundary(g) == NodeId[]
    end

    @testset "compactify does not mutate the original network" begin
        g = TensorNetwork()
        n = add_node!(g)
        w = add_wire!(g, 2; label=:σ)
        A = QTensor(randn(2, 2), (upper(:σ, 2), lower(:τ, 2)))
        ℓ = add_leg!(g, A, n, w)
        attach!(g, ℓ)
        compactify(g)
        @test attachment(g.wires[w]) === HalfLoose()   # still half-loose on the ORIGINAL
        @test length(nodes(g)) == 1                     # no outer node leaked into g itself
    end
end

@testset "Topograph: is_ordinary and is_closed" begin
    @testset "is_ordinary is true when compactify introduces no circles" begin
        # A half-loose wire (an open physical leg, as in an OBC-MPS) is not itself a circle, and compactifying it does not introduce one either.
        g = TensorNetwork()
        n = add_node!(g)
        w = add_wire!(g, 2; label=:σ)
        A = QTensor(randn(2, 2), (upper(:σ, 2), lower(:τ, 2)))
        ℓ = add_leg!(g, A, n, w)
        attach!(g, ℓ)
        @test attachment(g.wires[w]) === HalfLoose()
        @test is_ordinary(g)
    end

    @testset "is_ordinary is false when a circle is present" begin
        g = TensorNetwork()
        add_wire!(g, 5; label=:loop, closed=true)
        @test !is_ordinary(g)
    end

    @testset "is_closed is true only when every wire is already pinned" begin
        g, w, ℓA, ℓB = mk_pair(lower(:vL, 4), upper(:vL, 4))
        pin!(g, ℓA, ℓB)
        @test is_closed(g)
    end

    @testset "is_closed is false when a half-loose leg remains, e.g. an OBC-MPS physical leg" begin
        g = TensorNetwork()
        n = add_node!(g)
        w = add_wire!(g, 2; label=:σ)
        A = QTensor(randn(2, 2), (upper(:σ, 2), lower(:τ, 2)))
        ℓ = add_leg!(g, A, n, w)
        attach!(g, ℓ)
        @test attachment(g.wires[w]) === HalfLoose()
        @test !is_closed(g)
    end
end
