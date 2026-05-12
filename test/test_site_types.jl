using HalfIntegers
using TensorKit: TensorKit

@testset "Site types" begin
    # ── Site structs: SpinSite, FermionicSite, BosonicSite ───────────────────

    @testset "SpinSite" begin

        ## spin-1/2
        s = SpinSite(half(1), 1)
        @test s.spin_quantum_number == half(1)
        @test s.lattice_ordinal == 1
        @test local_hilbert_dim(s) == 2 # 2S+1 = 2*(1/2)+1 = 2

        ## spin-1/2 with U(1) symmetry (conserve Sz)
        s_u1 = SpinSite(half(1), 1; symmetry=:U1)
        @test s_u1.space == TensorKit.U1Space(half(1) => 1, -half(1) => 1) # spin up sector Sz=+1/2, spin down sector Sz=-1/2
        @test local_hilbert_dim(s_u1) == 2

        ## spin-1/2 with SU(2) symmetry (conserve full rotational symmetry)
        s_su2 = SpinSite(half(1), 1; symmetry=:SU2) # total spin = 1/2
        @test s_su2.space == TensorKit.SU2Space(half(1) => 1) # sector with total spin =1/2
        @test local_hilbert_dim(s_su2) == 2

        ## spin-1
        s1 = SpinSite(half(2), 2)
        @test local_hilbert_dim(s1) == 3 #2S+1 = 2*(1)+1 = 3 => -1,0,1

        ## no symmetry — plain ℂ², no block structure
        s_none = SpinSite(half(1), 1; symmetry=:none)
        @test s_none.space == TensorKit.ComplexSpace(2)  # unstructured ℂ², useful for debugging or symmetry-free Hamiltonians

        # validation tests
        @test_throws ArgumentError SpinSite(half(-1), 1) # 2S+1 would be 0 or negative, giving an empty or nonsensical Hilbert space
    end

    @testset "SpinlessFermionicSite" begin
        f = SpinlessFermionicSite(3)
        @test f.lattice_ordinal == 3
        @test local_hilbert_dim(f) == 2 # 2 possible states: occupied/unoccupied

        ## With U(1) - conserve total particle number.[Ĥ,N̂]=0 --> systems remains in fixed N sector blocks.
        f_u1 = SpinlessFermionicSite(2; symmetry=:U1)
        @test f_u1.space == TensorKit.U1Space(0 => 1, 1 => 1) # occupied/unoccupied single site
        @test local_hilbert_dim(f_u1) == 2

        ## With Z2 - conserve parity (whether total number of particles even or odd) but might not conserve the total number N. Eg: superconductors with Cooper pairs
        f_z2 = SpinlessFermionicSite(1; symmetry=:Z2)
        @test f_z2.space == TensorKit.Z2Space(0 => 1, 1 => 1)  # even parity/odd parity sectors
        @test local_hilbert_dim(f_z2) == 2

        ## no symmetry — plain ℂ², useful for Hamiltonians with no conserved quantities
        f_none = SpinlessFermionicSite(1; symmetry=:none)
        @test f_none.space == TensorKit.ComplexSpace(2)
        @test local_hilbert_dim(f_none) == 2
    end

    @testset "SpinlessHardCoreBosonicSite" begin # b† |0⟩ = |1⟩, b† |1⟩ = 0
        # at most 1 per site - similar to 2 state basis like fermions but obeying bosonic statistics
        a = SpinlessHardCoreBosonicSite(4)
        @test a.lattice_ordinal == 4
        @test local_hilbert_dim(a) == 2 # hard-core-> 2 possible states: occupied/unoccupied

        ## With U(1) - conserve total particle number.[Ĥ,N̂]=0 --> systems remains in fixed N sector blocks
        a_u1 = SpinlessHardCoreBosonicSite(2; symmetry=:U1)
        @test a_u1.space == TensorKit.U1Space(0 => 1, 1 => 1) # occupied/unoccupied single site
        @test local_hilbert_dim(a_u1) == 2

        ## no symmetry — plain ℂ², useful for debugging or symmetry-free Hamiltonians
        a_none = SpinlessHardCoreBosonicSite(1; symmetry=:none)
        @test a_none.space == TensorKit.ComplexSpace(2)
        @test local_hilbert_dim(a_none) == 2
    end

    @testset "SpinlessBosonicSite" begin # b† |n⟩ = √(n+1) |n+1⟩
        b = SpinlessBosonicSite(1; n_max_occ=3)
        @test b.lattice_ordinal == 1
        @test b.n_max_occ == 3
        @test local_hilbert_dim(b) == 4 # n_max_occ + 1

        ## With U(1) - conserve total particle number.[Ĥ,N̂]=0 --> systems remains in fixed N sector blocks
        b_u1 = SpinlessBosonicSite(1; n_max_occ=3, symmetry=:U1)
        @test b_u1.n_max_occ == 3
        @test local_hilbert_dim(b_u1) == 4 # n_max_occ + 1
        @test b_u1.space == TensorKit.U1Space(0 => 1, 1 => 1, 2 => 1, 3 => 1) # different particle number sectors

        @test_throws ArgumentError SpinlessBosonicSite(1; n_max_occ=-1)  # negative occupation meaningless

        ## no symmetry — plain ℂⁿ, useful for debugging or symmetry-free Hamiltonians
        b_none = SpinlessBosonicSite(1; n_max_occ=3, symmetry=:none)
        @test b_none.space == TensorKit.ComplexSpace(4)  # dim = n_max_occ + 1 = 4
        @test local_hilbert_dim(b_none) == 4
    end
end