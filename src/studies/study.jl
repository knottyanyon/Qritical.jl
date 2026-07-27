# study regimes

"""
    StudyType

Root of the **study** axis: *which regime of physical properties is under investigation*. A study names the **regime** a run works in, and the regime fixes which class of *target properties* is accessible, so it sets the stage for the run and decides which algorithms and targets are meaningful. It is, by design, neither an algorithm, which says how to compute, nor a protocol, which says on what schedule; those are separate, orthogonal axes of `solve`. The two regimes are separated mathematically rather than by vocabulary. A [`StaticsStudy`](@ref) poses an eigenproblem: it solves for a fixed state and then measures operators on it. A [`DynamicsStudy`](@ref) poses an initial-value problem: it prepares a state and propagates it, reading how observables change with time. The split is sharp because some quantities live in only one regime. As an example, consider the spectral gap ``\\Delta = E_1 - E_0``. It is a well-posed static quantity, yet it is meaningless for a single propagating state. Conversely, the way a local observable ``\\expval{O(t)}`` evolves after a sudden quench is intrinsically dynamical, and undefined for a fixed eigenstate.

Encoding the regime as a **type**, rather than a boolean flag, is what lets one `solve` driver serve both regimes by dispatch while keeping the regime an explicit, checkable part of a run's specification.

See also: [`StaticsStudy`](@ref), [`DynamicsStudy`](@ref).
"""
abstract type StudyType end   # Julia's `abstract type` keyword declares an abstract supertype; no objects of this type can be instantiated directly; concrete subtypes (or sub-abstract types) inherit from it using `<:`; physics: the "study regime" axis of the solve() interface

"""
    StaticsStudy <: StudyType

The **static** regime: the object of interest is a property of a state you obtain by solving for it directly, not by evolving it through time. That state is the solution of an eigenproblem or fixed-point problem, for instance a ground state, an excited level, or a spectral gap, and operators are then measured on it.
"Static" here means time-independence, or stationarity, which is *not* the same as thermodynamic equilibrium: a state can be perfectly stationary yet sit far from equilibrium. Describing the regime as non-propagating simply means no equation of motion is integrated. There is no time argument to advance, because the quantities of interest are properties of the fixed solution itself rather than of any history it passed through.
"""
abstract type StaticsStudy <: StudyType end

"""
    DynamicsStudy <: StudyType

The **dynamic** regime: the object of interest is how a state or observable evolves in time under a Hamiltonian, and is time-ordered when ``H`` itself varies in time. How the Hamiltonian is tuned along the way, that is, the perturbation schedule, is the **protocol**'s business and not the study's.
"""
abstract type DynamicsStudy <: StudyType end
