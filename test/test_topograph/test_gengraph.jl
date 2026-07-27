"""
Tests for: AbstractGenTopoGraph, GraphTrait/Ungraded/Oriented/Polarised (src/topograph/gengraph.jl).
"""

# A throwaway level-0-only graph type, exercising the trait default without needing a real LatticeGraph. It never overrides graph_trait, so it must fall back to Ungraded().
struct DummyLevel0Graph <: AbstractGenTopoGraph end

@testset "Topograph: AbstractGenTopoGraph and its traits" begin
    @testset "TensorNetwork is an AbstractGenTopoGraph, tagged Polarised" begin
        @test TensorNetwork <: AbstractGenTopoGraph
        @test graph_trait(TensorNetwork) === Polarised()
    end

    @testset "a type that never overrides graph_trait defaults to Ungraded" begin
        @test graph_trait(DummyLevel0Graph) === Ungraded()
    end

    @testset "is_oriented/is_polarised read the tower correctly for TensorNetwork" begin
        g = TensorNetwork()
        @test is_oriented(g)
        @test is_polarised(g)
    end

    @testset "is_oriented/is_polarised are false for an Ungraded (level-0-only) type" begin
        # Polarised implies Oriented implies Ungraded's absence of guarantees, not the other way around: an Ungraded type gets neither of the stronger guarantees.
        @test !is_oriented(DummyLevel0Graph())
        @test !is_polarised(DummyLevel0Graph())
    end
end
