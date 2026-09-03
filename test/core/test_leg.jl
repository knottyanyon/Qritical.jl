@testitem "PenroseLabel" begin
    a = PenroseLabel(:A)
    a2 = PenroseLabel(:A, 2)
    b = PenroseLabel(:B)

    @test a == PenroseLabel(:A)
    @test a != a2
    @test a != b
    @test hash(a) == hash(PenroseLabel(:A))

    ad = orientation_dual(a)
    @test ad isa PenroseLabel{Dual}
    @test orientation_dual(ad) == a
    @test a != ad   # different orientations are never equal, even with the same family/index
end

@testitem "Leg forwarding" begin
    import TensorKit: TensorKit   # narrow import: TensorKit also exports `dim`/`space`, which
    # would collide with Qritical's own bindings under a blanket `using TensorKit` here.

    ix = TIx(4)
    leg = Leg(PenroseLabel(:A), ix)

    @test dim(leg) == 4
    @test space(leg) == TensorKit.ComplexSpace(4)
    @test symmetry_structure(leg) == symmetry_structure(ix)
    @test entanglement_structure(leg) == entanglement_structure(ix)

    @test leg == Leg(PenroseLabel(:A), TIx(4))
    @test leg != Leg(PenroseLabel(:B), TIx(4))
end
