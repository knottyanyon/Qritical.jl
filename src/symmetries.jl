# §6.1  Symmetry tags — sectorless backend for now; future upgrades to graded spaces.
#
# Keeping symmetry-related types in a separate file makes the eventual TensorKit
# integration a single-file concern rather than a change scattered across dof.jl.

# ----------------------------------------------------------------------------------------
# Symmetry tags
# ----------------------------------------------------------------------------------------

"""
    NoSymmetry

Tag type selecting the sectorless (dense, symmetry-ignorant) backend for tensor
and operator construction.

Right now every physical space in Qritical.jl uses this backend: operators are
plain `ComplexF64` matrices and MPS bond spaces are ungraded `Int` dimensions.

In future this will be replaced by a graded `ElementarySpace` from TensorKit.jl
that carries quantum-number sectors (e.g. ``U(1)`` particle number or
``SU(2)`` spin).  The tag is introduced now so that every function signature
that will eventually dispatch on symmetry already has a slot for it; switching
backends will then only require adding new methods.

See also: [`physical_space`](@ref)
"""
struct NoSymmetry end

"""
    physical_space(dof::AbstractDoF, ::NoSymmetry) -> Int

Return an integer representing the local physical space of `dof` in the
sectorless backend.

Currently this simply returns [`local_dim(dof)`](@ref) — a plain `Int` that
the MPS/MPO constructors use as the physical-leg dimension.

When symmetry support is added in the future, this function will
gain new methods dispatching on symmetry tags such as `U1Symmetry()` and will
return a graded `ElementarySpace` from TensorKit.jl carrying the full
quantum-number sector structure.  All Hamiltonian and MPS construction code
should call `physical_space` rather than `local_dim` so it automatically
benefits from that upgrade.

# Examples

```jldoctest
julia> physical_space(SpinHalf(), NoSymmetry())
2

julia> physical_space(Electron(), NoSymmetry())
4
```
"""
physical_space(dof::AbstractDoF, ::NoSymmetry) = local_dim(dof)
