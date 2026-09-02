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

# legs carry an index
# when indices are repartitioned and grouped, it is the leg that carries the information of partitioning and repartitioning
# so we don't need the Multix as a separate struct anymore

# code design idea stubs

# when we start with tensors with information about symmetries (the required quantum numbers that we need to conserve) then the TIx belongs to a physical leg
# if it doesn't (which corresponds to trivial sectors for domain and codomain in TensorKit) then it is just a leg

# symmetry obeyed by the local hilbert space 
# for a specific state, the relevant question for symmetric-tensor purposes is: which symmetry operations map this particular state back to itself (up to phase)? That's a much smaller group in general. this means we need to find the "stabilizer group" of that particular state. and this "stabilizer group" is that much smaller sub group of the hilbert space symmetry group

abstract type AbstractStructure end

abstract type SymmetryStructure <: AbstractStructure end

abstract type Origin end
abstract type Classical <: Origin end
abstract type Quantum <: Origin end

abstract type CorrelationsStructure <: AbstractStructure end

abstract type EntanglementStructure <: AbstractStructure end

abstract type AbstractLeg end

## structure encoded
# the density operator encodes correlation structure among other things
