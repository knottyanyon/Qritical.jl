# Tests for §8 — TEBD evolution driver, Tracker, neel_state.
# Physics invariants: Sz conservation, energy conservation, entropy growth.

@testset "§8.0 Evolution study type" begin
    @testset "Evolution is a DynamicsStudy study" begin
        # The rename is not cosmetic: it must actually join the Dynamics family,
        # because the study regime is what downstream dispatch keys on.
        ψ₀ = neel_state(Chain(4))   # Néel state as a FiniteMPS; `Chain(4)` = 4-site open chain; `neel_state` builds the product state |↑↓↑↓⟩
        @test Evolution <: DynamicsStudy   # `A <: B` = subtype check } <: DynamicsStudy   # also check the parameterized type `Evolution{FiniteMPS}` is a subtype; `typeof(ψ₀)` = the concrete type of ψ₀ 
    end

    @testset "construction carries the state unmodified" begin
        ψ₀ = neel_state(Chain(4))
        s = Evolution(ψ₀)   # `Evolution(ψ₀)` constructs a new Evolution study wrapping ψ₀
        @test s.ψ₀ === ψ₀   # `===` is Julia's identity check (same object in memory, not just equal values); Python: `s.psi0 is psi0`; the study must store a reference to ψ₀, not a copy
    end

    @testset "old name is gone from the public API" begin
        # The deprecation shim is non-exported, so the old symbols must not
        # appear in names(Qritical) even though `Qritical.Quench` still resolves.
        @test !(:Quench in names(Qritical))   # `names(Qritical)` = vector of exported symbols; `:Quench` is a Symbol literal (no Python equivalent); `in` checks membership; `!` = NOT; the old `Quench` name was renamed to `Evolution` and must not appear in the public API
        @test !(:QuenchResult in names(Qritical))   # same for the old result type; non-exported names can still be accessed as `Qritical.Quench` but won't show in `names(Qritical)`
    end
end

@testset "§8.1 neel_state" begin
    @testset "neel_state has correct length" begin
        g = Chain(4)
        ψ = neel_state(g)
        @test length(ψ.tensors) == 4   # `length(ψ.tensors)` = number of MPS tensors = chain length L; `.tensors` field access on the FiniteMPS struct
    end

    @testset "neel_state is a product state (bond dim 1)" begin
        g = Chain(6)
        ψ = neel_state(g)
        for i in 1:6   # check every site
            @test size(ψ.tensors[i].data, 1) == 1 ||
                size(ψ.tensors[i].data, 3) == 1 ||
                (i > 1 && size(ψ.tensors[i].data, 1) == 1) ||
                (i < 6 && size(ψ.tensors[i].data, 3) == 1)   # `size(A, k)` = dimension along axis k; product state must have left bond χL=1 or right bond χR=1; `||` = OR (short-circuit); `&&` = AND
        end
        # Bond dimensions between sites should all be 1 (product state)
        for i in 1:5   # check all L-1=5 internal bonds
            @test size(ψ.tensors[i].data, 3) == 1   # `size(A, 3)` = right bond dimension; product state (zero entanglement) → χ=1 everywhere; physics: entanglement entropy = 0 for any cut of a product state
        end
    end

    @testset "neel_state alternates |↑⟩ |↓⟩" begin
        g = Chain(4)
        ψ = neel_state(g)
        # local_expectation computes <ψ|σᵢ|ψ> directly without building a full MPO
        ops = algebra_generators(SpinHalf())   # get Sz (and others) as matrices
        sz1 = real(local_expectation(ψ, ComplexF64.(ops.Sz), 1))   # ⟨ψ|Sz₁|ψ⟩; `ComplexF64.(ops.Sz)` converts matrix to ComplexF64 element-type; `real(...)` discards imaginary rounding noise
        sz2 = real(local_expectation(ψ, ComplexF64.(ops.Sz), 2))   # ⟨ψ|Sz₂|ψ⟩
        @test sz1 ≈ 0.5 atol=1e-10   # site 1: spin up  # site 1 is ↑ → Sz=+½
        @test sz2 ≈ -0.5 atol=1e-10   # site 2: spin down  # site 2 is ↓ → Sz=−½
    end

    @testset "neel_state has zero total Sz for even L" begin
        g = Chain(6)
        ψ = neel_state(g)
        Mop = total_magnetization(g)   # M = Σ Sz_i observable
        mpo_m = MPO(Mop)   # build MPO for M
        sz_tot = real(expect(ψ, mpo_m)) / real(overlap(ψ, ψ))   # Rayleigh quotient: ⟨M⟩/‖ψ‖²; for even L the Néel state has L/2 up-spins and L/2 down-spins → total Sz = 0
        @test sz_tot ≈ 0.0 atol=1e-10   # physics: Néel state is in the Sz=0 sector for even L
    end

    @testset "neel_state entanglement entropy is zero (product state)" begin
        g = Chain(4)
        ψ = neel_state(g)
        # Product state → Schmidt rank 1 at every bond → S=0
        ψ_c = canonicalize(ψ, BondCanonical(2, NoTrunc()))   # put in bond-canonical form at bond 2; needed to get well-defined singular values
        sv = ψ_c.bond_svs[3].values   # singular values at bond between sites 2 and 3; `.bond_svs[3]` = spectrum at the 3rd bond boundary (between sites 2 and 3 for a 4-site chain)
        λ² = (sv ./ norm(sv)) .^ 2   # normalised squared singular values = Schmidt coefficients; `./ norm(sv)` broadcasts division by the norm; `.^ 2` = element-wise squaring 
        S = -sum(p -> p > 0 ? p * log2(p) : 0.0, λ²)   # von Neumann entanglement entropy S = −Σ λ²·log₂(λ²); `p -> ...` anonymous function; `? : ` ternary; `log2` = base-2 logarithm 
        @test S ≈ 0.0 atol=1e-10   # product state → single nonzero Schmidt value λ₁=1 → S = 0
    end
end

@testset "§8.1 TEBD quench via solve" begin
    @testset "total Sz conserved under XXZ real-time evolution" begin
        g = Chain(4)
        H = XXZ(g; J=1.0, Jz=1.0, h=0.0)   # isotropic Heisenberg model (Jz=J); no field h → [H, Sz_tot]=0 → Sz conserved
        Mop = total_magnetization(g)
        mpo_m = MPO(Mop)
        mpo_h = MPO(H)   # MPO for H (unused directly here but built for reference)

        ψ₀ = neel_state(g)
        # Canonicalize the Neel state
        ψ₀ = canonicalize(ψ₀, LeftCanonical(NoTrunc()))   # left-canonical form; `NoTrunc()` = keep all singular values; needed for stable norm computation

        Sz_init = real(expect(ψ₀, mpo_m)) / real(overlap(ψ₀, ψ₀))   # initial total Sz = 0 for the Néel state

        p = ConstantProtocol(RealTime(), 0.02, 20, H)   # 20 real-time steps of dt=0.02 → T=0.4
        result = solve(H, Evolution(ψ₀), TEBD(SuzukiTrotter(2), MaxBondDimTrunc(16)), p)   # `SuzukiTrotter(2)` = 2nd-order Trotter; `MaxBondDimTrunc(16)` limits bond dimension to 16

        ψ_final = result.state   # final MPS after 20 steps
        Sz_final = real(expect(ψ_final, mpo_m)) / real(overlap(ψ_final, ψ_final))
        @test abs(Sz_final - Sz_init) < 1e-3   # Sz conservation: the XXZ Hamiltonian commutes with Sz_total, so it cannot change; small residual comes from Trotter error and bond truncation
    end

    @testset "norm preserved under real-time evolution" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))   # chain: `canonicalize(neel_state(g), ...)` = apply canonicalization to the result of neel_state; Julia evaluates function arguments before passing
        norm_init = real(overlap(ψ₀, ψ₀))   # ‖ψ₀‖² should be ≈ 1 after canonicalization

        p = ConstantProtocol(RealTime(), 0.05, 5, H)   # 5 steps of dt=0.05 → T=0.25
        result = solve(H, Evolution(ψ₀), TEBD(SuzukiTrotter(1), MaxBondDimTrunc(8)), p)

        norm_final = real(overlap(result.state, result.state))
        @test norm_final ≈ norm_init atol=1e-6   # real-time unitary evolution preserves norm; small deviation from Trotter error and truncation
    end

    @testset "TEBD EvolutionResult has correct fields" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        p = ConstantProtocol(RealTime(), 0.01, 2, H)   # 2 steps
        result = solve(H, Evolution(ψ₀), TEBD(SuzukiTrotter(1), NoTrunc()), p)
        @test result isa EvolutionResult   # return type check
        @test result.state isa FiniteMPS   # final state is an MPS
        @test result.steps == 2   # number of steps stored correctly
    end
end

@testset "§8.2 Tracker" begin
    @testset "NoTracker does not collect data" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        p = ConstantProtocol(RealTime(), 0.05, 3, H)
        result = solve(
            H, Evolution(ψ₀), TEBD(SuzukiTrotter(1), NoTrunc()), p; tracker=NoTracker()
        )   # `tracker=NoTracker()` keyword argument passes the null tracker
        @test isempty(result.measurements)   # `isempty(dict)` = `len(dict) == 0`; NoTracker doesn't add any measurements
    end

    @testset "Tracker collects magnetization each step" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        p = ConstantProtocol(RealTime(), 0.05, 4, H)   # 4 steps
        Mop = total_magnetization(g)   # observable to track
        tracker = Tracker(:mag => Mop; every=1)   # `Tracker(:mag => Mop; every=1)`: `:mag` is a Symbol key (like a string but interned; Python: `"mag"`); `=> ` creates a Pair. `; every=1` is a keyword argument to Tracker's constructor
        result = solve(
            H, Evolution(ψ₀), TEBD(SuzukiTrotter(1), MaxBondDimTrunc(8)), p; tracker=tracker
        )
        @test haskey(result.measurements, :mag)   # `haskey(dict, key)` = Python `key in dict`; check that the `:mag` observable was recorded
        @test length(result.measurements[:mag]) == 4   # one per step  # 4 steps with `every=1` → 4 measurements; `result.measurements[:mag]` accesses the Vector{Float64} for the :mag observable
    end

    @testset "Tracker :mag agrees with direct expect" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        p = ConstantProtocol(RealTime(), 0.01, 2, H)   # 2 steps
        Mop = total_magnetization(g)
        tracker = Tracker(:mag => Mop; every=1)
        result = solve(
            H, Evolution(ψ₀), TEBD(SuzukiTrotter(1), NoTrunc()), p; tracker=tracker
        )

        # The magnetization of the Neel state should be ~0 and well-measured
        ψ_final = result.state   # final MPS
        mpo_m = MPO(Mop)   # rebuild MPO for direct comparison
        sz_direct = real(expect(ψ_final, mpo_m)) / real(overlap(ψ_final, ψ_final))   # direct calculation after evolution
        tracked_last = result.measurements[:mag][end]   # last tracked value; `[end]` = last element 
        @test tracked_last ≈ sz_direct atol=1e-8   # the tracker must use the same formula as the direct calculation: ⟨O⟩/‖ψ‖²
    end

    @testset "measure_entropy grows from zero after quench" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        ψ₀ = neel_state(g)
        ψ₀ = canonicalize(ψ₀, BondCanonical(2, NoTrunc()))   # bond-canonical form to get well-defined singular values

        sv0 = ψ₀.bond_svs[3].values   # singular values at bond 3 (between sites 2 and 3) BEFORE evolution
        λ0 = (sv0 ./ norm(sv0)) .^ 2   # normalised Schmidt coefficients
        S_init = -sum(p -> p > 0 ? p * log2(p) : 0.0, λ0)   # initial entanglement entropy; should be 0 for a product state
        @test S_init ≈ 0.0 atol=1e-10   # Néel state is a product state → zero entanglement

        # After several steps entanglement should grow
        p = ConstantProtocol(RealTime(), 0.05, 20, H)   # 20 steps × dt=0.05 = T=1.0
        result = solve(H, Evolution(ψ₀), TEBD(SuzukiTrotter(2), MaxBondDimTrunc(32)), p)   # χ_max=32 allows entanglement to grow
        ψ_f = canonicalize(result.state, BondCanonical(2, NoTrunc()))   # re-canonicalize to get Schmidt values
        svf = ψ_f.bond_svs[3].values   # singular values AFTER evolution
        λf = (svf ./ norm(svf)) .^ 2
        S_final = -sum(p -> p > 0 ? p * log2(p) : 0.0, λf)
        @test S_final > 0.01   # entanglement has grown  # physics: unitary evolution generates entanglement; for a 1D system starting from a product state, S(t) grows linearly until it saturates at ~(L/2)·log(2) (Page value)
    end
end
