@testset "§6.1/§9.2 LatticeGraph — the second AbstractGenTopoGraph implementor" begin

    @testset "type hierarchy" begin
        @test LatticeGraph <: UndirectedGraph
    end

    @testset "level-0 only: neither Oriented() nor Polarised()" begin
        @test graph_trait(LatticeGraph) === Ungraded()
        g = LatticeGraph(Chain(4))
        @test !is_oriented(g)
        @test !is_polarised(g)
    end

    @testset "nodes match the layout's sites" begin
        g = LatticeGraph(Chain(4))
        @test nodes(g) == collect(sites(Chain(4)))
    end

    @testset "ends(g, ℓ) is an unordered pair, no start/finish on a lattice bond" begin
        g = LatticeGraph(Chain(4))
        ℓ = first(links(g))
        e = ends(g, ℓ)
        @test e isa Set{Int}
        @test length(e) == 2
    end

    @testset "attachment(g, ℓ) === Pinned() for every edge, OBC and PBC alike" begin
        for g in (LatticeGraph(Chain(4)), LatticeGraph(Chain(4, true)))
            for ℓ in links(g)
                @test attachment(g, ℓ) === Pinned()
            end
        end
    end

    @testset "incident / degree" begin
        g = LatticeGraph(Chain(4))
        @test degree(g, 1) == 1   # OBC end site: one bond, (1,2)
        @test degree(g, 2) == 2   # bulk site: two bonds, (1,2) and (2,3)
        @test length(incident(g, 2)) == 2

        g_pbc = LatticeGraph(Chain(4, true))
        @test degree(g_pbc, 1) == 2   # PBC: every site has two bonds
    end

    @testset "boundary is a node predicate, derived from degree" begin
        @test boundary(LatticeGraph(Chain(4))) == [1, 4]
        @test boundary(LatticeGraph(Chain(4, true))) == []
    end

    @testset "compactify is the identity on an ordinary graph" begin
        g = LatticeGraph(Chain(4))
        @test compactify(g) === g
        @test boundary(compactify(g)) == [1, 4]
    end

    @testset "a Chain generates a TensorNetwork whose nodes match its sites" begin
        g = Chain(4)
        net = TensorNetwork()
        for _ in sites(g)
            add_node!(net)
        end
        @test length(nodes(net)) == length(sites(g))
    end

end
