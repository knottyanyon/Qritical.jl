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

# Code glossary for `src/operations/` - see `docs/src/dev/kitchen_sink.md`'s "Using Code
# Glossaries" section. Terms defined here are reused by that submodule's docstrings only
# (currently `trotter_gates.jl`).

Glossaries.@define!(:trotter_norm, :name, "norm")
Glossaries.@define!(:trotter_norm, :type, "Function")
Glossaries.@define!(
    :trotter_norm,
    :description,
    "a caller-supplied callback `group -> Real` returning one commuting group's aggregate operator norm - forwarded as-is to `Subroutines.local_error_bound`/`Subroutines.accumulate_trotter_error!`"
)

Glossaries.@define!(:trotter_hamiltonian, :name, "hamiltonian")
Glossaries.@define!(:trotter_hamiltonian, :type, "Hamiltonian")
Glossaries.@define!(
    :trotter_hamiltonian,
    :description,
    "the same `Hamiltonian` a `TrotterStep` was built from - re-grouped internally via `split_commuting_groups` so the error bound mirrors exactly what `trotterize` grouped, rather than trusting a caller to pass matching pre-grouped terms"
)
