#=META
source:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
docstrings:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
refs:
credits: N/A
=#

"""
    Hamiltonian

A specialized [`Observable`](@ref): the generator of time evolution via a [`Propagator`](@ref).
Currently a placeholder - no fields yet, pending further design - but concrete and instantiable
(`Hamiltonian()`), unlike the abstract [`Observable`](@ref) it subtypes.
"""
struct Hamiltonian <: Observable end

"""
    to_mpo(::Hamiltonian) -> MPOperator

Materialize a [`Hamiltonian`](@ref) into an `MPOperator` - a new method on the existing generic
`to_mpo` (dispatched specifically on `::Hamiltonian`, not the abstract [`Observable`](@ref): a
`Hamiltonian` must always become an MPO to build the [`Propagator`](@ref) it generates, unlike
[`Correlator`](@ref), whose typical two-point measurements don't need this path at all - no
`to_mpo(::Correlator)` method exists). **Not yet implemented** - depends on `Hamiltonian` exposing
a real tensor/term representation, which doesn't exist yet.
"""
function to_mpo(::Hamiltonian)
    return error(
        "to_mpo(::Hamiltonian) is not yet implemented - Hamiltonian has no tensor/term " *
        "representation yet",
    )
end
