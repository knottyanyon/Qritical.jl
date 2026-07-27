@testset "§6.1 OrdinaryGraphNetwork, UndirectedGraph — the ordinary-graph branch" begin

    @testset "type hierarchy" begin
        @test OrdinaryGraphNetwork <: AbstractGenTopoGraph
        @test UndirectedGraph <: OrdinaryGraphNetwork
    end

end
