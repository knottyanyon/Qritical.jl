# `import`/method-extension of a `using`-imported generic function only works at true top-level
# module scope in Julia, not inside a local scope like a `@testset`/`@testitem` body (which the
# runtests.jl shim maps `@testitem` to) - so this test-only collector and its `step!`/`finalize!`
# extensions live at the top of the file, outside any @testitem.
import Qritical: step!, finalize!

struct EchoCollector <: Qritical.AbstractCollector
    seen::Vector{Any}
end
EchoCollector() = EchoCollector(Any[])
step!(::Qritical.Active, c::EchoCollector, ctx::NamedTuple) = push!(c.seen, ctx)
finalize!(::Qritical.Active, c::EchoCollector) = c.seen

@testitem "RecordingTrait / AbstractCollector" begin
    @test RecordingTrait(NoOpCollector()) isa Inactive
    @test step!(NoOpCollector(), (; a=1)) === nothing
    @test finalize!(NoOpCollector()) === nothing

    c = EchoCollector()
    @test RecordingTrait(c) isa Active
    step!(c, (; bond=1))
    step!(c, (; bond=2))
    @test finalize!(c) == [(; bond=1), (; bond=2)]
end
