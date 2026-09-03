#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs:
credits: N/A
=#

"""
    Hamiltonian

A specialized [`Observable`](@ref): the generator of time evolution via a [`Propagator`](@ref).
`H = Σⱼ cⱼ · Oⱼ`, written as a sum of multipartite operators over an `L`-site chain - the same
data `Subroutines.build_automaton` (`src/subroutines/automaton.jl`) needs to materialize an MPO,
so `Hamiltonian`'s fields mirror `build_automaton`'s arguments exactly with no translation layer.

# Fields

  - `terms           :: Vector{AutomatonTerm}` - the sum-of-multipartite-operators decomposition.
  - `L               :: Int` - chain length.
  - `physical_spaces :: Union{TensorKit.ElementarySpace,Vector{<:TensorKit.ElementarySpace}}` -
    single space (broadcast to every site) or one space per site.

`tensor`/`outputs`/`inputs` remain [`Observable`](@ref)'s abstract-dispatched stub errors -
unaffected by these fields: a `Hamiltonian` is not reducible to one `QProcess`/tensor except by
materializing via [`to_mpo`](@ref) (which produces an `MPOperator`, itself many per-site
`QProcess`es, not one tensor).
"""
struct Hamiltonian <: Observable
    terms::Vector{AutomatonTerm}
    L::Int
    physical_spaces::Union{TensorKit.ElementarySpace,Vector{<:TensorKit.ElementarySpace}}
end

"""
    to_mpo(H::Hamiltonian) -> MPOperator

Materialize `H` into a genuinely closed `MPOperator` via `Subroutines.build_automaton`/
`materialize`/`close_transducer_boundary` - a new method on the existing generic `to_mpo`
(dispatched specifically on `::Hamiltonian`, not the abstract [`Observable`](@ref): a `Hamiltonian`
must always become an MPO to build the [`Propagator`](@ref) it generates, unlike [`Correlator`](@ref),
whose typical two-point measurements don't need this path at all - no `to_mpo(::Correlator)` method
exists).

`materialize`'s raw output is a weighted transducer with non-trivial boundary bonds (see
`close_transducer_boundary`'s docstring); `to_mpo` always closes it down to a genuine operator with
trivial dim-1 boundary bonds, since every consumer of `to_mpo(H)` (`apply`, `evaluate_expectation_value`,
...) needs a closed operator, not the raw automaton.
"""
function to_mpo(H::Hamiltonian)
    automaton = build_automaton(H.terms, H.L, H.physical_spaces)
    return close_transducer_boundary(automaton, materialize(automaton))
end

"""
    symmetry_group(H::Hamiltonian) -> Type{<:TensorKit.Sector}

The symmetry group `H`'s Hilbert space is graded by, read directly off `H.physical_spaces` via
`TensorKit.sectortype` - the group every term's operators are built to respect, not a
per-operator introspection (in this codebase's `QProcess` formalism, an operator's leg space
already *is* the sectortype source, so inspecting individual operator tensors would be
redundant with this).
"""
function symmetry_group(H::Hamiltonian)
    space = H.physical_spaces isa AbstractVector ? H.physical_spaces[1] : H.physical_spaces
    return TensorKit.sectortype(space)
end
