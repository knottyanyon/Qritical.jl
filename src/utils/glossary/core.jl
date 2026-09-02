#=META
source:
  author: Bavithra
  coauthor:
  reviewer:
docstrings:
  author: Bavithra
  coauthor:
  reviewer:
refs:
credits:
=#

# Code glossary for `src/core/` - see `docs/src/dev/kitchen_sink.md`'s "Using Code
# Glossaries" section. Terms defined here are reused by docstrings across `structure_traits.jl`
# and `tix.jl`; a term lives here (not in `_core.jl`) because it was first needed by this theme.

Glossaries.@define!(:space, :name, "space")
Glossaries.@define!(:space, :type, "TensorKit.ElementarySpace")
Glossaries.@define!(
    :space,
    :description,
    "the leg's underlying vector space, carrying its full symmetry-sector structure - trivial `ComplexSpace` for the unsymmetric case, or a graded space"
)

Glossaries.@define!(:leg, :name, "leg")
Glossaries.@define!(:leg, :type, "LegRole")
Glossaries.@define!(
    :leg,
    :description,
    "the role this leg plays in a tensor network ([`PhysicalLeg`](@ref) or [`VirtualLeg`](@ref))"
)

Glossaries.@define!(:penrose_family, :name, "family")
Glossaries.@define!(:penrose_family, :type, "Symbol")
Glossaries.@define!(:penrose_family, :description, "the wire's base name (`:A`, `:B`, ...)")

Glossaries.@define!(:penrose_index, :name, "index")
Glossaries.@define!(:penrose_index, :type, "Int")
Glossaries.@define!(
    :penrose_index,
    :description,
    "`0` for a bare wire (`A`); `1`, `2`, `3`, ... for an enumerated family member (`A_1`, `A_2`, `A_3`, ...)"
)

Glossaries.@define!(:penrose_label, :name, "label")
Glossaries.@define!(:penrose_label, :type, "PenroseLabel")
Glossaries.@define!(:penrose_label, :description, "the wire's abstract, named identity")

Glossaries.@define!(:bridging_ix, :name, "ix")
Glossaries.@define!(:bridging_ix, :type, "AbstractIx")
Glossaries.@define!(
    :bridging_ix,
    :description,
    "the [`TIx`](@ref) (or [`MulTIx`](@ref)) bridging `label` to a concrete `TensorKit.ElementarySpace`"
)
