@testset "IndexedTensor" begin

    @testset "round-trip: data is preserved exactly" begin
        data = [1.0 2.0; 3.0 4.0]
        i = TIx{Upper}(:i, 2)
        j = TIx{Lower}(:j, 2)
        t = IndexedTensor(data, (i, j))
        @test t.data == data
    end

    @testset "index count matches array rank" begin
        data = rand(2, 3, 4)
        indices = (
            TIx{Upper}(:i, 2),
            TIx{Lower}(:j, 3),
            TIx{Upper}(:k, 4),
        )
        t = IndexedTensor(data, indices)
        @test length(t.indices) == ndims(t.data)
        @test length(t.indices) == 3
    end

    @testset "AbstractArray interface is delegated" begin
        data = rand(Float64, 3, 2)
        i = TIx{Upper}(:i, 3)
        j = TIx{Lower}(:j, 2)
        t = IndexedTensor(data, (i, j))
        @test size(t) == (3, 2)
        @test t[1, 2] == data[1, 2]
        @test ndims(t) == 2
    end

    @testset "order-0 scalar: empty index tuple and 0-dim array" begin
        data = fill(42.0)
        t = IndexedTensor(data, ())
        @test ndims(t.data) == 0
        @test length(t.indices) == 0
        @test t.data[] == 42.0
    end

end
