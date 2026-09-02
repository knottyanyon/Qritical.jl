#=META
source:
  author: Bavithra
  coauthor:
  reviewer:
docstrings:
  author: Bavithra
  coauthor:
  reviewer:
refs: simon_2023
credits:
=#

# Code glossary for `src/models/` — see `docs/src/dev/kitchen_sink.md`'s "Using Code
# Glossaries" section. Terms defined here are reused by docstrings across `algebra_tags.jl`
# and `dof.jl`; a term lives here (not in `_core.jl`) because it was first needed by this theme.

Glossaries.@define!(:dof, :name, "dof")
Glossaries.@define!(:dof, :type, "AbstractDoF")
Glossaries.@define!(
    :dof,
    :description,
    "the degree-of-freedom instance carrying the compile-time symmetry/statistics tag for one lattice site"
)

Glossaries.@define!(:group_order, :name, "GroupOrder")
Glossaries.@define!(:group_order, :type, "Int")
Glossaries.@define!(
    :group_order,
    :description,
    "the \$N\$ in \$SU(N)\$ (2 for spin, 3 for color, ...), distinct from the Lie algebra's own dimension \$N^2-1\$"
)

Glossaries.@define!(:dynkin_label, :name, "TableauLabel")
Glossaries.@define!(:dynkin_label, :type, "Union{Int, NTuple{N,Int}}")
Glossaries.@define!(
    :dynkin_label,
    :description,
    "the Tableau label(s) uniquely picking out an irreducible representation of \$SU(\\mathrm{GroupOrder})\$: a bare `Int` at \$N=2\$, an `NTuple{N-1,Int}` at \$N \\geq 3\$"
)

Glossaries.@define!(:reference_operator, :name, "reference_operator")
Glossaries.@define!(:reference_operator, :type, "TensorKit.AbstractTensorMap")
Glossaries.@define!(
    :reference_operator,
    :description,
    "a `TensorMap` built by `TensorKitTensors`' own reference implementation, graded by that package's sector type rather than this `dof`'s `compile_site_space`"
)

Glossaries.@define!(:relabel, :name, "relabel")
Glossaries.@define!(:relabel, :type, "Function")
Glossaries.@define!(
    :relabel,
    :description,
    "maps each of `reference_operator`'s `TensorKitSectors.blocksectors` to the equivalent sector of this `dof`'s own `compile_site_space`"
)
