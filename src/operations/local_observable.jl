#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs:
credits: N/A
=#

"""
    LocalObservable

A specialized [`Observable`](@ref): one or more single-physical-leg operators measured at named
sites, contracted directly against a state without ever building an `MPOperator` - the kind of
cheap measurement (magnetization, staggered magnetization, a two-point spin correlator) a
completed TEBD/DMRG run would evaluate. Named `LocalObservable`, not `Correlator` - "correlator"
specifically means a correlation between ≥2 separated points, and a single spin operator at one
site (`LocalObservable(i, sz)`) is not a correlator; both are simply the `n=1`/`n>1` case of the
same "measure these operators at these sites" shape.

Fields mirror [`AutomatonTerm`](@ref)'s own `ops::Vector{Pair{Int,QProcess}}` site+operator
bundling convention exactly. Unlike `AutomatonTerm`/TEBD gates, sites are **not** required to be
contiguous - this is a measurement, not an evolution gate, so a long-range two-point correlator
like `⟨Ŝᶻ₁Ŝᶻ₁₀⟩` is exactly the kind of thing this type represents.

# Fields

  - `ops :: Vector{Pair{Int,QProcess}}` - `(site, single-physical-leg-endomorphism)` pairs.
"""
struct LocalObservable <: Observable
    ops::Vector{Pair{Int,QProcess}}
end

"""
    LocalObservable(site::Int, op::QProcess)

Single-site convenience constructor - the common case of measuring one spin operator at one site,
without wrapping it in a vector by hand.
"""
LocalObservable(site::Int, op::QProcess) = LocalObservable([site => op])
