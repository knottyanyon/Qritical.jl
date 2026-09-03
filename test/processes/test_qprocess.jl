@testitem "QProcess: wiring TIx legs to a TensorKit TensorMap" begin
    using TensorKit

    @testitem "codomain_legs and domain_legs" begin
        using TensorKit

        V = ComplexSpace(2)

        @testitem "default role is PhysicalLeg" begin
            using TensorKit
            V = ComplexSpace(2)
            t = zeros(ComplexF64, V ⊗ V ← one(V))
            @test codomain_legs(t) == (TIx(V), TIx(V))
            @test domain_legs(zeros(ComplexF64, one(V) ← V)) == (TIx(V),)
        end

        @testitem "custom single role broadcasts to every leg" begin
            using TensorKit
            V = ComplexSpace(2)
            t = zeros(ComplexF64, V ⊗ V ← one(V))
            @test codomain_legs(t; roles=VirtualLeg()) ==
                (TIx(V, VirtualLeg()), TIx(V, VirtualLeg()))
        end

        @testitem "custom per-leg role tuple" begin
            using TensorKit
            V = ComplexSpace(2)
            t = zeros(ComplexF64, V ⊗ V ← one(V))
            legs = codomain_legs(t; roles=(PhysicalLeg(), VirtualLeg()))
            @test legs == (TIx(V, PhysicalLeg()), TIx(V, VirtualLeg()))
        end

        @testitem "mismatched role-count throws ArgumentError" begin
            using TensorKit
            V = ComplexSpace(2)
            t = zeros(ComplexF64, V ⊗ V ← one(V))
            @test_throws ArgumentError codomain_legs(t; roles=(PhysicalLeg(),))
            @test_throws ArgumentError domain_legs(
                zeros(ComplexF64, one(V) ← V); roles=(PhysicalLeg(), VirtualLeg())
            )
        end

        @testitem "symmetric (GradedSpace) legs" begin
            using TensorKit
            g = GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1)
            t = zeros(ComplexF64, g ← one(g))
            @test codomain_legs(t) == (TIx(g),)
        end
    end

    @testitem "QProcess construction" begin
        using TensorKit

        @testitem "default-role construction from a TensorMap" begin
            using TensorKit
            V = ComplexSpace(2)
            t = zeros(ComplexF64, V ← V)
            p = QProcess(t)
            @test tensor(p) === t
            @test outputs(p) == (TIx(V),)
            @test inputs(p) == (TIx(V),)
        end

        @testitem "leg-count mismatch throws DimensionMismatch" begin
            using TensorKit
            V = ComplexSpace(2)
            t = zeros(ComplexF64, V ← V)
            @test_throws DimensionMismatch QProcess(t, (TIx(V), TIx(V)), (TIx(V),))
            @test_throws DimensionMismatch QProcess(t, (TIx(V),), ())
        end

        @testitem "leg-space mismatch throws ArgumentError" begin
            using TensorKit
            V = ComplexSpace(2)
            W = ComplexSpace(3)
            t = zeros(ComplexF64, V ← V)
            @test_throws ArgumentError QProcess(t, (TIx(W),), (TIx(V),))
            @test_throws ArgumentError QProcess(t, (TIx(V),), (TIx(W),))
        end

        @testitem "symmetric-sector QProcess" begin
            using TensorKit
            g = GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1)
            t = zeros(ComplexF64, g ← g)
            p = QProcess(t)
            @test outputs(p) == (TIx(g),)
            @test inputs(p) == (TIx(g),)
        end
    end

    @testitem "State: a process with no inputs" begin
        using TensorKit

        @testitem "construction and predicates" begin
            using TensorKit
            V = ComplexSpace(2)
            ψ = zeros(ComplexF64, V ← one(V))
            ψ[1] = 1
            s = State(ψ)
            @test is_state(s)
            @test !is_effect(s)
            @test outputs(s) == (TIx(V),)
            @test inputs(s) == ()
        end

        @testitem "nonzero domain throws ArgumentError" begin
            using TensorKit
            V = ComplexSpace(2)
            t = zeros(ComplexF64, V ← V)   # numin(t) == 1, not a valid State
            @test_throws ArgumentError State(t)
        end

        @testitem "custom leg role" begin
            using TensorKit
            V = ComplexSpace(2)
            ψ = zeros(ComplexF64, V ← one(V))
            s = State(ψ; roles=VirtualLeg())
            @test outputs(s) == (TIx(V, VirtualLeg()),)
        end
    end

    @testitem "Effect: a process with no outputs" begin
        using TensorKit

        @testitem "construction and predicates (symmetric sector)" begin
            using TensorKit
            g = GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1)
            π = zeros(ComplexF64, one(g) ← g)
            e = Effect(π)
            @test is_effect(e)
            @test !is_state(e)
            @test inputs(e) == (TIx(g),)
            @test outputs(e) == ()
        end

        @testitem "nonzero codomain throws ArgumentError" begin
            using TensorKit
            V = ComplexSpace(2)
            t = zeros(ComplexF64, V ← V)   # numout(t) == 1, not a valid Effect
            @test_throws ArgumentError Effect(t)
        end
    end

    @testitem "Scalar: a process with neither inputs nor outputs" begin
        using TensorKit

        @testitem "value extracts the scalar" begin
            using TensorKit
            V = ComplexSpace(2)
            t = TensorMap(fill(3.0 + 1.0im), one(V), one(V))
            sc = Scalar(t)
            @test value(sc) == 3.0 + 1.0im
            @test is_state(sc)
            @test is_effect(sc)
        end

        @testitem "nonzero legs is a MethodError, not a runtime check" begin
            # Scalar's type parameter pins N₁ == N₂ == 0 directly on AbstractTensorMap - a
            # wrong-shaped tensor fails via Julia's own dispatch, before any Scalar-specific
            # code runs at all. This documents that guarantee rather than a thrown ArgumentError.
            using TensorKit
            V = ComplexSpace(2)
            t = zeros(ComplexF64, V ← V)
            @test_throws MethodError Scalar(t)
        end
    end

    @testitem "equal_up_to_scalar" begin
        using TensorKit

        @testitem "a nonzero-scalar multiple of itself is true" begin
            using TensorKit
            V = ComplexSpace(2)
            ψ = zeros(ComplexF64, V ← one(V))
            ψ[1] = 1
            ψ[2] = 2
            s1 = State(ψ)
            s2 = State(3 * ψ)
            @test equal_up_to_scalar(s1, s2)
        end

        @testitem "an unrelated (non-parallel) tensor of the same shape is false" begin
            using TensorKit
            V = ComplexSpace(2)
            ψ = zeros(ComplexF64, V ← one(V))
            ψ[1] = 1
            φ = zeros(ComplexF64, V ← one(V))
            φ[2] = 1
            @test !equal_up_to_scalar(State(ψ), State(φ))
        end

        @testitem "mismatched leg spaces returns false rather than throwing" begin
            using TensorKit
            V2 = ComplexSpace(2)
            V3 = ComplexSpace(3)
            ψ2 = zeros(ComplexF64, V2 ← one(V2))
            ψ2[1] = 1
            ψ3 = zeros(ComplexF64, V3 ← one(V3))
            ψ3[1] = 1
            @test !equal_up_to_scalar(State(ψ2), State(ψ3))
        end

        @testitem "both-zero case is true; one-zero-one-nonzero case is false" begin
            using TensorKit
            V = ComplexSpace(2)
            zero_ψ = zeros(ComplexF64, V ← one(V))
            other_zero_ψ = zeros(ComplexF64, V ← one(V))
            @test equal_up_to_scalar(State(zero_ψ), State(other_zero_ψ))

            nonzero_ψ = zeros(ComplexF64, V ← one(V))
            nonzero_ψ[1] = 1
            @test !equal_up_to_scalar(State(zero_ψ), State(nonzero_ψ))
        end
    end
end
