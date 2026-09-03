#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs: schollwoeck_2011
credits: N/A
=#

# Code glossary for `src/algorithms/` - see `docs/src/dev/kitchen_sink.md`'s "Using Code
# Glossaries" section. Terms defined here are reused by `tebd.jl`'s docstrings only.

Glossaries.@define!(:tebd_hamiltonian, :name, "hamiltonian")
Glossaries.@define!(:tebd_hamiltonian, :type, "Hamiltonian")
Glossaries.@define!(:tebd_hamiltonian, :description, "the generator of time evolution")

Glossaries.@define!(:tebd_state, :name, "state")
Glossaries.@define!(:tebd_state, :type, "MPState")
Glossaries.@define!(:tebd_state, :description, "the initial state to evolve")

Glossaries.@define!(:tebd_dt, :name, "dt")
Glossaries.@define!(:tebd_dt, :type, "Float64")
Glossaries.@define!(
    :tebd_dt, :description, "the (real- or imaginary-)time step per outer step"
)
