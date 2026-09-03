@testitem "RecordingTrait / AbstractErrorAccumulator" begin
    @test RecordingTrait(NoOpErrorAccumulator()) isa Inactive
    @test record!(NoOpErrorAccumulator(), (; ε=0.1)) === nothing
    @test finalize!(NoOpErrorAccumulator()) === nothing
end

@testitem "QuadratureTruncationErrorAccumulator" begin
    acc = QuadratureTruncationErrorAccumulator()
    @test RecordingTrait(acc) isa Active
    record!(acc, (; ε=0.3))
    record!(acc, (; ε=0.4))
    @test finalize!(acc) ≈ hypot(0.3, 0.4)
    @test acc.history == [0.3, 0.4]
    @test finalize!(acc; nrm=2.0) ≈ hypot(0.3, 0.4) / 2.0
end
