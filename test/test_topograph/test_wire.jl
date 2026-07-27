"""
Tests for: Wire (src/topograph/wire.jl).
"""

using TensorKit: TensorKit

@testset "Topograph: Wire owns the space" begin   # Wire holds the shared physical data (space, spectrum); Leg (tested separately) holds only the per-end variance
    @testset "constructor defaults: unattached, open, unlabelled-dof" begin
        w = Wire(WireId(1), 4; label=:vL)   # `Wire(id, space; label=...)` = keyword constructor; `4` here plays the role of `S` in `Wire{S}` a bare Int dim, standing in for a graded ElementarySpace later; `;` separates positional from keyword args
        @test w.id == WireId(1)      # field access `w.id` confirms the id round-trips
        @test w.space == 4           # `w.space` is AUTHORITATIVE. the dim/space lives here, not on either Leg
        @test w.start === nothing    # a freshly constructed Wire has NEITHER end attached
        @test w.finish === nothing   # both ends default to `nothing`, meaning
        @test w.closed == false      # this Wire is not a Circle
        @test w.label == :vL
        @test w.dof === nothing      # `dof === nothing` ⟺ this is a virtual bond (no physical Hilbert space attached), per the Wire docstring's `Union{Nothing,AbstractDoF}` field
        @test w.spectrum === nothing # no Schmidt spectrum recorded here yet either
    end

    @testset "a spectrum may only live on a virtual bond (dof === nothing)" begin
        # A Wire with both a dof and a spectrum cannot be constructed. Schmidt spectra live on cuts, cuts are virtual bonds, physical legs carry a dof never both at once. This is "one line, catches a class of bugs" per the design doc.
        fake_dof = SpinHalf()   # `SpinHalf()` = an existing concrete AbstractDoF from dof.jl; stands in for "this wire carries a physical leg"
        fake_spectrum = SingValSpectrum([1.0], 0.0, true)   # `SingValSpectrum(values, ε, normalized)` = existing type from spectrum.jl; stands in for "this wire is a cut with recorded singular values"
        @test_throws ArgumentError Wire(
            WireId(1), 2; label=:σ, dof=fake_dof, spectrum=fake_spectrum
        )
    end

    @testset "space works with an ungraded TensorKit ElementarySpace, not just a bare Int" begin
        # ComplexSpace carries no symmetry sectors, it is TensorKit's version of a plain dimension, but it is a real ElementarySpace, not an Int.
        V = TensorKit.ComplexSpace(4)
        w = Wire(WireId(1), V; label=:vL)
        @test w isa Wire{TensorKit.ComplexSpace}   # S is inferred from V, same mechanism as the Int case
        @test w.space == V               # space is stored verbatim, no coercion to Int anywhere
        @test attachment(w) === Loose()  # attachment only ever looks at start/finish/closed, so it must not care what S is
    end

    @testset "space works with a U(1) graded ElementarySpace" begin
        # A graded space is the actually interesting case: it carries charge sectors (here 0, +1, -1 with their multiplicities), which is exactly what a physical leg with a conserved quantum number needs Wire.space to be able to hold.
        G = TensorKit.Vect[TensorKit.U1Irrep](0 => 1, 1 => 2, -1 => 2)
        w = Wire(WireId(1), G; label=:σ)
        @test w isa Wire{typeof(G)}
        @test w.space == G
        @test TensorKit.dim(w.space) == 5   # TensorKit's dim sums sector dimension × multiplicity, here 1 + 2 + 2
        @test attachment(w) === Loose()
    end

    @testset "the dof/spectrum invariant does not depend on what S is" begin
        # Repeats the mutual-exclusion check above with a graded space, to confirm the constructor's guard is about (dof, spectrum), never about the space type.
        G = TensorKit.Vect[TensorKit.U1Irrep](0 => 1, 1 => 2, -1 => 2)
        fake_dof = SpinHalf()
        fake_spectrum = SingValSpectrum([1.0], 0.0, true)
        @test_throws ArgumentError Wire(
            WireId(1), G; label=:σ, dof=fake_dof, spectrum=fake_spectrum
        )
    end
end
