using Test   # `using Test` imports `@test`, `@testset`, `@test_throws`, etc.; Julia's built-in testing macros 
using Qritical   # `using Qritical` imports all exported names from the Qritical package; this is needed to access `StudyType`, `StaticsStudy`, `DynamicsStudy`

@testset "study hierarchy" begin   # top-level test group for the study type hierarchy
    @test StaticsStudy <: StudyType   # `A <: B` tests whether type A is a subtype of B ; physics: StaticsStudy must be part of the StudyType hierarchy so `solve` can dispatch on it
    @test DynamicsStudy <: StudyType   # same check for DynamicsStudy; both regimes must be subtypes of the root

    @test isabstracttype(StudyType)   # `isabstracttype(T)` returns true if T cannot be instantiated directly ; StudyType is declared as `abstract type StudyType end` — no objects of this type can exist
    @test isabstracttype(StaticsStudy)   # StaticsStudy is also abstract; only its concrete children (e.g. `GroundState`) can be instantiated
    @test isabstracttype(DynamicsStudy)   # DynamicsStudy is abstract; only `Evolution` (its concrete subtype) can be instantiated
    # we rely heavily on this during dispatch make sure nothing gets mixed
    @testset "regimes are disjoint" begin   # nested testset; Julia allows arbitrary nesting of @testsets for hierarchical reporting
        @test !(StaticsStudy <: DynamicsStudy)   # `!(A <: B)` = NOT subtype; regimes must be disjoint — a type cannot be both static and dynamic at the same time; physics: the two regimes are orthogonal: eigenproblems vs initial-value problems
        @test !(DynamicsStudy <: StaticsStudy)   # also check the reverse direction; neither inherits from the other
    end
end

@testset "exports" begin
    @test all(s -> s in names(Qritical), (:StudyType, :StaticsStudy, :DynamicsStudy))   # `names(Module)` returns the list of exported names from a module ` checks all elements satisfy pred; checks that the public API exports all three study types
end
