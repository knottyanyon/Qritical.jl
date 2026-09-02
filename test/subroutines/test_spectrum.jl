@testitem "SingValSpectrum" begin
    s = SingValSpectrum([0.8, 0.6], 0.0, true)
    @test length(s) == 2
    @test schmidt_rank(s) == 2
    @test spectral_gap(s) ≈ 0.2
    @test local_truncation_error(s) == 0.0
    @test local_truncation_error(0.05) == 0.05

    s1 = SingValSpectrum([1.0], 0.1, false)
    @test spectral_gap(s1) ≈ 1.0

    p = entanglement_entropy(SingValSpectrum([1 / sqrt(2), 1 / sqrt(2)], 0.0, true))
    @test p ≈ 1.0   # maximally entangled qubit pair: 1 bit of entropy

    es = entanglement_spectrum(SingValSpectrum([1.0], 0.0, true))
    @test es ≈ [0.0]   # -2*ln(1) = 0

    @test global_truncation_error([0.1, 0.2]) ≈ hypot(0.1, 0.2)
    @test global_truncation_error([0.1, 0.2]; nrm=2.0) ≈ hypot(0.1, 0.2) / 2.0
end
