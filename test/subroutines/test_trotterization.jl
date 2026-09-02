@testitem "sequence(::LieTrotter, ...)" begin
    terms = [:A, :B, :C]
    seq = sequence(LieTrotter(), terms, 0.1)
    @test seq == [(:A, 1.0), (:B, 1.0), (:C, 1.0)]
end

@testitem "sequence(::SuzukiTrotter, ...)" begin
    terms = [:A, :B, :C]
    seq = sequence(SuzukiTrotter(), terms, 0.1)
    @test seq == [(:A, 0.5), (:B, 0.5), (:C, 1.0), (:B, 0.5), (:A, 0.5)]

    seq2 = sequence(SuzukiTrotter(), [:A, :B], 0.1)
    @test seq2 == [(:A, 0.5), (:B, 1.0), (:A, 0.5)]

    seq1 = sequence(SuzukiTrotter(), [:A], 0.1)
    @test seq1 == [(:A, 1.0)]
end

@testitem "sequence(::Suzuki4th, ...) recursive construction" begin
    terms = [:A, :B]
    dt = 0.1
    s = 1 / (4 - 4^(1 / 3))

    expected = vcat(
        [(term, c * s) for (term, c) in sequence(SuzukiTrotter(), terms, dt)],
        [(term, c * s) for (term, c) in sequence(SuzukiTrotter(), terms, dt)],
        [(term, c * (1 - 4s)) for (term, c) in sequence(SuzukiTrotter(), terms, dt)],
        [(term, c * s) for (term, c) in sequence(SuzukiTrotter(), terms, dt)],
        [(term, c * s) for (term, c) in sequence(SuzukiTrotter(), terms, dt)],
    )

    seq = sequence(Suzuki4th(), terms, dt)
    @test seq == expected
    @test length(seq) == 15

    # the 5 nested-S2 rescaling factors sum to 1 (s + s + (1-4s) + s + s == 1)
    @test s + s + (1 - 4s) + s + s ≈ 1.0
end

@testitem "TrotterErrorAccumulator RecordingTrait" begin
    @test RecordingTrait(TrotterErrorAccumulator()) isa Active
end

@testitem "TrotterErrorAccumulator plain-sum combination" begin
    acc = TrotterErrorAccumulator()
    record!(acc, (; step=1, local_error=0.1))
    record!(acc, (; step=2, local_error=0.2))
    @test finalize!(acc) ≈ 0.3
    @test acc.history == [0.1, 0.2]
end

@testitem "local_error_bound per-order scaling" begin
    terms = [1.0, 2.0]  # dummy terms; norm is identity on plain numbers
    dt = 0.1
    norm = identity
    N = sum(norm, terms)

    @test local_error_bound(LieTrotter(), terms, dt, norm) ≈ (N * dt)^2
    @test local_error_bound(SuzukiTrotter(), terms, dt, norm) ≈ (N * dt)^3 / 12
    @test local_error_bound(Suzuki4th(), terms, dt, norm) ≈ (N * dt)^5
end

@testitem "accumulate_trotter_error! sums num_steps identical local bounds" begin
    terms = [1.0, 2.0]
    dt = 0.1
    norm = identity
    num_steps = 5

    acc = TrotterErrorAccumulator()
    total = accumulate_trotter_error!(acc, SuzukiTrotter(), terms, dt, num_steps, norm)

    expected_per_step = local_error_bound(SuzukiTrotter(), terms, dt, norm)
    @test total ≈ num_steps * expected_per_step
    @test length(acc.history) == num_steps
end
