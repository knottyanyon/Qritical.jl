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
credits:
=#

# Code glossary for `src/processes/` - see `docs/src/dev/kitchen_sink.md`'s "Using Code
# Glossaries" section. Terms defined here are reused by docstrings across `qprocess.jl`; a term
# lives here (not in `_core.jl`) because it was first needed by this theme.

Glossaries.@define!(:tensor, :name, "tensor")
Glossaries.@define!(:tensor, :type, "TensorKit.AbstractTensorMap")
Glossaries.@define!(
    :tensor,
    :description,
    "the numerical representation of the process in a chosen basis - what TensorKit calls a linear map"
)

Glossaries.@define!(:roles, :name, "roles")
Glossaries.@define!(:roles, :type, "Union{LegRole, Tuple{Vararg{LegRole}}}")
Glossaries.@define!(
    :roles,
    :description,
    "the [`LegRole`](@ref) assigned to each leg being wired up - a single `LegRole` broadcasts to every leg, or a tuple matches leg-for-leg"
)
