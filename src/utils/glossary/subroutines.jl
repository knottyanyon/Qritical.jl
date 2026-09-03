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

# Code glossary for `src/subroutines/` - see `docs/src/dev/kitchen_sink.md`'s "Using Code
# Glossaries" section. Terms defined here are reused by docstrings across `spectrum.jl`,
# `decompositions.jl`, and `canonical_decompositions.jl`.

Glossaries.@define!(:sv_values, :name, "values")
Glossaries.@define!(:sv_values, :type, "AbstractVector{<:Real}")
Glossaries.@define!(
    :sv_values, :description, "descending singular values σ₁ ≥ σ₂ ≥ ... > 0"
)

Glossaries.@define!(:sv_epsilon, :name, "ε")
Glossaries.@define!(:sv_epsilon, :type, "Float64")
Glossaries.@define!(
    :sv_epsilon,
    :description,
    "Frobenius norm of the discarded singular values (`0.0` if untruncated)"
)

Glossaries.@define!(:sv_normalized, :name, "normalized")
Glossaries.@define!(:sv_normalized, :type, "Bool")
Glossaries.@define!(:sv_normalized, :description, "`true` iff `sum(values.^2) ≈ 1`")

Glossaries.@define!(:bond_cutoff, :name, "bond_cutoff")
Glossaries.@define!(:bond_cutoff, :type, "Union{Int, Nothing}")
Glossaries.@define!(
    :bond_cutoff,
    :description,
    "an optional integer bond dimension: when given, translated to `MatrixAlgebraKit.truncrank(bond_cutoff)` (keep the `bond_cutoff` largest singular values); when `nothing` (the default), no truncation is applied"
)
