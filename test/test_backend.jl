using HalfIntegers: half

@testset "Backend context" begin
    @testset "default backend is :native" begin
        @test current_backend() == :native
    end

    @testset "with_backend switches context" begin
        result = with_backend(:native) do
            current_backend()
        end
        @test result == :native
    end

    @testset "with_backend is scoped" begin
        @test current_backend() == :native
        with_backend(:tensorkit) do
            @test current_backend() == :tensorkit
        end
        @test current_backend() == :native
    end

    @testset "unknown backend throws" begin
        @test_throws ArgumentError with_backend(:unknown) do
        end
    end

    @testset ":tensorkit backend throws (not yet implemented)" begin
        @test_throws ErrorException with_backend(:tensorkit) do
            IndexedTensor(
                rand(2, 2),
                (
                    PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex),
                    BondIndex(:α, 1, 2, 2, DownIndex),
                ),
            )
        end
    end
end
