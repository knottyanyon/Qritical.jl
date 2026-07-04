@testset "§6.1 Symmetry tags — NoSymmetry / physical_space" begin

    @testset "physical_space — sectorless (NoSymmetry)" begin
        # For now: returns a plain integer dimension (sectorless ComplexSpace)
        @test physical_space(Spin{1//2}(),     NoSymmetry()) == 2
        @test physical_space(Spin{1}(),        NoSymmetry()) == 3
        @test physical_space(SpinlessFermion(), NoSymmetry()) == 2
        @test physical_space(Electron(),        NoSymmetry()) == 4
        @test physical_space(HardCoreBoson(),   NoSymmetry()) == 2
    end

end
