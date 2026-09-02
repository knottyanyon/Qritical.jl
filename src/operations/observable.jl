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
    Observable

Abstract root of the observable hierarchy: a physical observable, lazy - not yet materialized
into any tensor network (dense `QProcess`, `MPOperator`, or otherwise) until evaluated via
[`to_mpo`](@ref) (or whatever materialization path is eventually built). Julia requires an
abstract type here, not a concrete placeholder struct, because [`Hamiltonian`](@ref) and
[`Correlator`](@ref) need to subtype it directly (a concrete `struct Observable end` cannot itself
be subtyped) - so `Observable` is a category, not a directly-instantiable "empty object";
[`Hamiltonian`](@ref)/[`Correlator`](@ref) are its concrete members.

`tensor`/`outputs`/`inputs` (the [`AbstractProcess`](@ref) accessor trio every concrete subtype
must eventually implement) are stubbed here, dispatched on the abstract type, so any `Observable`
subtype gets a clear "not yet implemented" error for free until it has real fields to answer from.
"""
abstract type Observable <: AbstractProcess end

function tensor(::Observable)
    return error(
        "tensor(::Observable) is not yet implemented - Observable has no tensor representation yet",
    )
end
function outputs(::Observable)
    return error(
        "outputs(::Observable) is not yet implemented - Observable has no tensor representation yet",
    )
end
function inputs(::Observable)
    return error(
        "inputs(::Observable) is not yet implemented - Observable has no tensor representation yet",
    )
end
