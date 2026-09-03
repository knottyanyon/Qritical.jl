#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs: N/A
credits: N/A
=#

"""
    xxz_hamiltonian(L::Int, V::TensorKit.ElementarySpace; Jxy=1.0, Jz=1.0, h=0.0) -> Hamiltonian

The finite, open-boundary 1D spin-1/2 XXZ chain:

``H = \\sum_i J_{xy} \\cdot (\\hat{S}^x_i \\hat{S}^x_{i+1} + \\hat{S}^y_i \\hat{S}^y_{i+1}) + J_z \\cdot \\hat{S}^z_i \\hat{S}^z_{i+1} - h \\sum_i \\hat{S}^z_i``

(`Jxy=Jz` recovers the isotropic Heisenberg chain; `Jxy=0` recovers a classical Ising chain).
Built via [`AutomatonTerm`](@ref)s exactly as hand-built test Hamiltonians elsewhere in this
codebase, just packaged as a reusable function. Uses bare Pauli matrices (not `/2`-scaled spin
operators), matching the convention already established by this codebase's own hand-built test
Hamiltonians.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:xxz_L, :xxz_V]))

# Keywords

  - `Jxy` - the `XY` coupling.
  - `Jz`  - the `Z` coupling.
  - `h`   - the transverse field strength; `h=0` (the default) omits field terms entirely.
"""
function xxz_hamiltonian(
    L::Int, V::TensorKit.ElementarySpace; Jxy::Number=1.0, Jz::Number=1.0, h::Number=0.0
)
    sx = QProcess(
        TensorKit.TensorMap(ComplexF64[0 1; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    sy = QProcess(
        TensorKit.TensorMap(ComplexF64[0 -im; im 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    sz = QProcess(
        TensorKit.TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )

    terms = AutomatonTerm[]
    for i in 1:(L - 1)
        push!(terms, AutomatonTerm(Jxy, [i => sx, i + 1 => sx]))
        push!(terms, AutomatonTerm(Jxy, [i => sy, i + 1 => sy]))
        push!(terms, AutomatonTerm(Jz, [i => sz, i + 1 => sz]))
    end
    if h != 0
        for i in 1:L
            push!(terms, AutomatonTerm(-h, [i => sz]))
        end
    end
    return Hamiltonian(terms, L, V)
end

"""
    _product_state(L::Int, V::TensorKit.ElementarySpace, up::Function) -> MPState{LeftCanonical,FiniteSupport}

Build an exact, unentangled product state directly as trivial-bond site tensors - every `vL`/`vR`
bond carries the monoidal unit itself (via `TensorKit.insertleftunit`/`insertrightunit`, not a
hand-built dimension-1 space) - `up(i)::Bool` selects `|0⟩`/`|1⟩` at site `i`. Deliberately **not**
routed through a dense `d^L` tensor + [`to_mps`](@ref): a genuine product state's true Schmidt rank
is `1` at every bond, but a generic SVD-based decomposition of the padded `d^L` dense tensor
returns the full `min(d^i, d^{L-i})`-dimensional bond with the extra dimensions exactly zero - a
real, exactly rank-deficient degeneracy that later propagates into [`to_vidal`](@ref)/`apply_gate`'s
Vidal-form Schmidt-weight stripping. Building the trivial bonds directly sidesteps this entirely
(and is `O(L)`, not `O(d^L)`).
"""
function _product_state(L::Int, V::TensorKit.ElementarySpace, up::Function)
    sites = Vector{QProcess}(undef, L)
    for i in 1:L
        data = zeros(ComplexF64, TensorKit.dim(V))
        data[up(i) ? 1 : 2] = 1.0
        physical = TensorKit.TensorMap(data, V ← one(V))
        # Attach the monoidal unit as genuine vL/vR legs via TensorKit's own insertion machinery (`insertleftunit`/`insertrightunit` insert the unit object as a real leg; a hand-built
        with_vL = TensorKit.insertleftunit(physical, Val(1)) # (vL,σ) ← one(V)
        with_both = TensorKit.insertrightunit(with_vL, Val(2); dual=true) # (vL,σ,vR') ← one(V)
        t = TensorKit.permute(with_both, ((1, 2), (3,)))# (vL,σ) ← vR - `dual=true`
        # above makes the moved-to-domain leg land as a plain (non-dual) `vR` space, matching the domain-leg convention `to_mps`'s own SVD-based construction already uses (permuting a codomain leg into the domain otherwise dualizes it).
        sites[i] = QProcess(
            t; output_roles=(VirtualLeg(), PhysicalLeg()), input_roles=VirtualLeg()
        )
    end
    return MPState(sites, LeftCanonical(), L, L + 1, L, 0.0)
end

"""
    neel_state(L::Int, V::TensorKit.ElementarySpace) -> MPState{LeftCanonical,FiniteSupport}

The alternating product state `|↑↓↑↓...⟩` (`|0⟩` at odd sites, `|1⟩` at even sites, in the same
2-dimensional physical-leg convention [`xxz_hamiltonian`](@ref)'s `sz` uses) - a natural,
unentangled initial condition for a quench. See [`_product_state`](@ref) for why this is built
directly with trivial bond dimensions rather than via [`to_mps`](@ref).
"""
neel_state(L::Int, V::TensorKit.ElementarySpace) = _product_state(L, V, isodd)

"""
    domain_wall_state(L::Int, V::TensorKit.ElementarySpace) -> MPState{LeftCanonical,FiniteSupport}

The domain-wall product state `|↑↑...↑↓↓...↓⟩` (`|0⟩` on the left half, `|1⟩` on the right half, in
the same physical-leg convention [`neel_state`](@ref) uses) - another natural, unentangled initial
condition for a quench, built the same way.
"""
function domain_wall_state(L::Int, V::TensorKit.ElementarySpace)
    return _product_state(L, V, i -> i <= L ÷ 2)
end
