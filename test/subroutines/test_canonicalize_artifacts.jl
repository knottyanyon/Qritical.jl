# Smoke tests against real tutorial-data wavefunctions (not just random tensors), so the
# canonicalization pipeline is exercised on concrete, previously-generated states rather than
# only synthetic randn() inputs. Data lives in artifacts_data/tutorial_data/{psi,psi1,psi2}.jls -
# plain dense Float64 arrays, one axis per physical site (spin-1/2, local dim 2).
@testitem "to_mps / is_canonical / to_vidal on packaged artifact states" begin
    using TensorKit
    using Serialization: deserialize
    using LinearAlgebra: norm

    artifact_dir = joinpath(@__DIR__, "..", "..", "artifacts_data", "tutorial_data")

    for filename in ("psi.jls", "psi1.jls", "psi2.jls")
        arr = deserialize(joinpath(artifact_dir, filename))
        L = ndims(arr)
        d = size(arr, 1)
        @assert all(==(d), size(arr)) "expected a uniform local dimension in $filename"

        V = TensorKit.ComplexSpace(d)
        ψtensor = TensorKit.TensorMap(
            ComplexF64.(arr), reduce(⊗, ntuple(_ -> V, L)) ← one(V)
        )
        ψ = State(ψtensor)

        # left-canonical round trip
        chain = to_mps(ψ; form=:left)
        @test length(chain.sites) == L
        @test chain isa MPState{LeftCanonical,FiniteSupport}
        @test is_canonical(chain)
        @test is_gauge_fixed(chain)

        full = tensor(chain.sites[1])
        for i in 1:(L - 1)
            n = TensorKit.numind(full)
            full = TensorKit.permute(full, (Tuple(1:(n - 1)), (n,)))
            full = full * TensorKit.permute(tensor(chain.sites[i + 1]), ((1,), (2, 3)))
        end
        full = TensorKit.permute(full, (Tuple(1:TensorKit.numind(full)), ()))
        arr_rec = vec(convert(Array, full))
        arr_orig = vec(convert(Array, ψtensor)) / norm(ψtensor)
        phase = (arr_orig' * arr_rec) / abs(arr_orig' * arr_rec)
        @test norm(arr_rec ./ phase .- arr_orig) < 1e-8

        # right-canonical round trip
        rchain = to_mps(ψ; form=:right)
        @test rchain isa MPState{RightCanonical,FiniteSupport}
        @test is_canonical(rchain)

        # mixed-canonical re-gauging at an interior site
        k = L ÷ 2
        mchain = canonicalize(chain, SiteCanonicalize(k))
        @test mchain isa MPState{MixedCanonical,FiniteSupport}
        @test is_canonical(mchain)
        @test mchain.orthogonality_center == k

        # Vidal-form round trip
        Γs, λs = to_vidal(chain)
        gfull = tensor(Γs[1])
        for i in 1:(L - 1)
            n = TensorKit.numind(gfull)
            gfull = TensorKit.permute(gfull, (Tuple(1:(n - 1)), (n,)))
            gfull = gfull * λs[i]
            gfull = gfull * TensorKit.permute(tensor(Γs[i + 1]), ((1,), (2, 3)))
        end
        gfull = TensorKit.permute(gfull, (Tuple(1:TensorKit.numind(gfull)), ()))
        garr_rec = vec(convert(Array, gfull))
        gphase = (arr_orig' * garr_rec) / abs(arr_orig' * garr_rec)
        @test norm(garr_rec ./ gphase .- arr_orig) < 1e-8
    end
end
