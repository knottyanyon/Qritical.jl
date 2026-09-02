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
    Correlator

A specialized [`Observable`](@ref): a (single- or multi-site) expectation-value observable, e.g.
magnetization or staggered magnetization - the kind of measurement a completed TEBD/DMRG run
would evaluate against an evolved/optimized state. Currently a placeholder, same caveats as
[`Hamiltonian`](@ref): no fields yet, pending further design, but concrete and instantiable.

Deliberately kept a separate [`Observable`](@ref) subtype rather than folded into
[`Hamiltonian`](@ref)'s machinery: the two-point correlators typically of interest are simple
enough to evaluate directly against a state without ever building an `MPOperator` - unlike a
[`Hamiltonian`](@ref), which must be turned into an MPO to build the `Propagator` it generates
even when it's itself composed of simple two-body terms. `Correlator`'s eventual evaluation path
is therefore expected to bypass [`to_mpo`](@ref) rather than route through it - no
`to_mpo(::Correlator)` method exists - left open for the future design pass that gives
`Correlator` real fields.
"""
struct Correlator <: Observable end
