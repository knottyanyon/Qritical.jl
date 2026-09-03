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

# Code glossary for `src/simstudy/` - see `docs/src/dev/kitchen_sink.md`'s "Using Code
# Glossaries" section. Terms defined here are reused by docstrings across `collectors.jl` and
# `accumulators.jl`.

Glossaries.@define!(:ctx, :name, "ctx")
Glossaries.@define!(:ctx, :type, "NamedTuple")
Glossaries.@define!(
    :ctx,
    :description,
    "whatever the calling routine has on hand, bundled as a `NamedTuple` - a canonicalization sweep passes `(; direction, bond, spectrum)`, a time-stepper could pass `(; step_num, t, raw_data)`; the recorder only reads the fields its own method looks at"
)
