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
credits: N/A
=#

using TensorKit

# SECTION -  MPOperator-specific pieces: to_mpo / to_choi / to_operator (stubs)

# `TensorTrain`, `MPOperator`, `_site_roles(::Val{2})`, `is_canonical`, `canonicalize`, etc. are
# all defined generically in `canonical_decompositions.jl` and already work for `MPOperator`
# unchanged - nothing MPO-specific needs registering there. What's genuinely MPO-specific is
# *constructing* an `MPOperator` from a dense operator tensor, which needs a bra/ket <-> Choi
# (vectorized) leg conversion this pass intentionally defers (see design discussion): building
# it correctly requires an explicit fusion isomorphism between `σ_ket ⊗ σ_bra` and a single
# `d²`-dimensional leg (`TensorKit.fuse`/`TensorKit.isomorphism`), which is real work with its own
# correctness questions (symmetric-sector fusion, keeping `to_choi`/`to_operator` genuine mutual
# inverses) that deserves its own dedicated pass rather than being rushed in here.

"""
    to_choi(mpo_site::QProcess) -> QProcess

Convert a single `MPOperator` site tensor from operator-style storage (`(vL, σ_ket) | (vR, σ_bra)`)
into Choi/vectorized-style storage (a single combined `d²`-dimensional physical leg via the
Choi-Jamiołkowski isomorphism), for use as a first-class conversion utility independent of any
particular sweep. **Not yet implemented** - deferred pending the fusion-isomorphism machinery
(`TensorKit.fuse`/`TensorKit.isomorphism`) needed to combine `σ_ket ⊗ σ_bra` into one leg.
"""
function to_choi(mpo_site::QProcess)
    return error(
        "to_choi is not yet implemented - deferred pending the fusion-isomorphism machinery " *
        "needed to combine (σ_ket, σ_bra) into a single Choi-vectorized leg.",
    )
end

"""
    to_operator(choi_site::QProcess) -> QProcess

Inverse of [`to_choi`](@ref): convert a single Choi/vectorized-style site tensor (one combined
`d²`-dimensional physical leg) back into operator-style storage (`(vL, σ_ket) | (vR, σ_bra)`).
**Not yet implemented** - see [`to_choi`](@ref).
"""
function to_operator(choi_site::QProcess)
    return error(
        "to_operator is not yet implemented - see to_choi's docstring for what's deferred."
    )
end

"""
    to_mpo(Ô::QProcess; bond_cutoff=nothing, form::Symbol=:left,
           access=HasEntanglementSpectrum(), collector=SimStudy.NoOpCollector(),
           accumulator=SimStudy.NoOpErrorAccumulator()) -> MPOperator

Decompose a full dense operator `Ô` (an `L`-site operator `QProcess` with `L` ket outputs and `L`
bra inputs) into an [`MPOperator`](@ref), mirroring [`to_mps`](@ref)'s construction: vectorize
each site's `(σ_ket, σ_bra)` pair into a single Choi leg via [`to_choi`](@ref), reuse
[`orthonormalize`](@ref) unchanged on the resulting dense state-shaped tensor, then split each
resulting site back into operator-style storage via [`to_operator`](@ref).

**Not yet implemented** - depends on [`to_choi`](@ref)/[`to_operator`](@ref), deferred to a
follow-on pass alongside the rest of the Choi/`Operator{Tag}` builder work.
"""
function to_mpo(
    Ô::QProcess;
    bond_cutoff::Union{Int,Nothing}=nothing,
    form::Symbol=:left,
    access::AccessEntanglementSpectrumData=HasEntanglementSpectrum(),
    collector::AbstractCollector=NoOpCollector(),
    accumulator::AbstractErrorAccumulator=NoOpErrorAccumulator(),
)
    return error(
        "to_mpo is not yet implemented - depends on to_choi/to_operator, deferred to a " *
        "follow-on pass.",
    )
end
