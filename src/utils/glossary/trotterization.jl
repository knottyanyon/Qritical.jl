#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs:
credits: [mpskit-validation](../..) internal validation package
=#

# Code glossary for `src/subroutines/trotterization.jl` - see `docs/src/dev/kitchen_sink.md`'s
# "Using Code Glossaries" section. Terms defined here are reused by that file's docstrings only.

Glossaries.@define!(:pf_terms, :name, "terms")
Glossaries.@define!(:pf_terms, :type, "AbstractVector")
Glossaries.@define!(
    :pf_terms,
    :description,
    "an ordered, fully opaque list of the pieces a generator has been split into (e.g. odd/even bond terms of a Hamiltonian) - `sequence` never inspects, exponentiates, or otherwise touches an individual term, it only reorders and pairs them with coefficients"
)

Glossaries.@define!(:pf_dt, :name, "dt")
Glossaries.@define!(:pf_dt, :type, "Real")
Glossaries.@define!(
    :pf_dt, :description, "the timestep that each returned coefficient multiplies"
)

Glossaries.@define!(:pf_norm, :name, "norm")
Glossaries.@define!(:pf_norm, :type, "Function")
Glossaries.@define!(
    :pf_norm,
    :description,
    "a caller-supplied callback `term -> Real` returning one term's operator norm - the only numeric information [`local_error_bound`](@ref) needs from a term, used through the submultiplicative commutator bound `‖[A,B]‖ ≤ 2‖A‖‖B‖` rather than ever computing an actual commutator"
)
