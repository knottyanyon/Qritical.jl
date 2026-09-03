#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs: schollwoeck_2011
credits: mpskit (github.com/QuantumKitHub/MPSKit.jl) - fuser-isomorphism pattern for `Base.:*(::AbstractMPO, ::AbstractMPS)`
=#

# `apply`/`overlap`/`norm`: the missing tensor-contraction primitives per Schollwoeck's TEBD
# breakdown. Greenfield - no bond-leg-fusion helper existed anywhere in the codebase before this
# file (`to_choi`/`to_operator` in mpoperator.jl are deferred pending exactly this machinery).
# `apply` returns the exact, untruncated product (bond dim multiplies) - truncation stays a
# separate explicit `canonicalize(...; bond_cutoff=...)` step, matching both this codebase's own
# construction/truncation separation (`to_mps` then `canonicalize`) and MPSKit.jl's own
# `mpo*mps`/`changebonds!` separation. `overlap` sweeps a small boundary tensor across the chain
# (Schollwoeck's standard method) and never materializes a full-size intermediate object.

# SECTION -  apply - exact, untruncated MPOperator applied to an MPState

# Contract mpo's σ_bra leg (bulk site domain-leg 2, i.e. leg 4 of (vL,σ_ket)|(vR,σ_bra)) against
# mps's σ leg (bulk site codomain-leg 2, i.e. leg 2 of (vL,σ)|(vR)) - the one leg that must match
# exactly. Isolate each as the sole leg on its composing side so `*` contracts only that leg.
function _contract_physical(t_mpo, t_mps)
    mpo_iso = TensorKit.permute(t_mpo, ((1, 2, 3), (4,)))   # (vL_o,σ_ket,vR_o) ← σ_bra
    mps_iso = TensorKit.permute(t_mps, ((2,), (1, 3)))       # σ ← (vL_s,vR_s)
    return mpo_iso * mps_iso   # (vL_o,σ_ket,vR_o) ← (vL_s,vR_s)
end

# Fuse the (vL_o,vL_s) leg pair of `raw` (Step A's output) into one vL_new leg via the isomorphism
# `Fl : vL_new ← (vL_o,vL_s)`, keeping σ_ket/vR_o/vR_s untouched. Returns vL_new ← (σ_ket,vR_o,vR_s).
function _fuse_left(raw, Fl)
    raw2 = TensorKit.permute(raw, ((1, 4), (2, 3, 5)))   # (vL_o,vL_s) ← (σ_ket,vR_o,vR_s)
    return Fl * raw2   # vL_new ← (σ_ket,vR_o,vR_s)
end

# Fuse the (vR_o,vR_s) leg pair of `step1` (post `_fuse_left`) into one vR_new leg via the
# isomorphism `Fr : vR_new ← (vR_o,vR_s)`, producing a legit bulk MPState site (vL_new,σ_ket)←(vR_new).
function _fuse_right(step1, Fr)
    step2 = TensorKit.permute(step1, ((1, 2), (3, 4)))   # (vL_new,σ_ket) ← (vR_o,vR_s)
    return step2 * TensorKit.adjoint(Fr)   # (vL_new,σ_ket) ← vR_new
end

function _bond_fuser(storagetype, Vo, Vs)
    return TensorKit.isomorphism(storagetype, TensorKit.fuse(Vo, Vs), Vo ⊗ Vs)
end

"""
    ApplyStrategy

Abstract root of `apply`'s dispatch trait, mirroring [`CanonicalizeConfig`](@ref)'s existing
pattern: `apply` needs two genuinely different procedures depending on what the caller wants next.
Concrete subtypes: [`ExactApply`](@ref) (default), [`CompressedApply`](@ref).
"""
abstract type ApplyStrategy end

"""
    ExactApply()

`apply` strategy: the exact, untruncated product (bond dim exactly `D_mpo*D_mps` at every bond).
The right choice whenever the enlarged state is immediately consumed further (e.g.
`evaluate_expectation_value`'s `overlap(state, apply(to_mpo(H), state))` never keeps the enlarged
state around, so compressing it would be wasted work). See [`ApplyStrategy`](@ref).
"""
struct ExactApply <: ApplyStrategy end

"""
    CompressedApply(bond_cutoff=nothing)

`apply` strategy: Schollwoeck §5.3's fused procedure - bring `mpo`/`mps` into matching canonical
form *first* (checked via `is_canonical`, not performed here), then SVD-truncate each bond
*while multiplying out on the fly*, never materializing the full untruncated bond at any point.
The right choice for TEBD, which needs exactly this every step. See [`ApplyStrategy`](@ref).
"""
struct CompressedApply <: ApplyStrategy
    bond_cutoff::Union{Int,Nothing}
end
CompressedApply() = CompressedApply(nothing)

# Two-stage per-site contraction shared by both ExactApply and CompressedApply: contract physical
# legs, then fuse the surviving vL/vR virtual-leg pairs into single new bond legs via isomorphism.
# Returns (result, Fr) - Fr (this site's right-bond fuser) is what the caller reuses as the *next*
# site's Fl, which is what keeps the resulting bond legs correctly identified across the chain.
function _apply_site(t_mpo, t_mps, Fl, storagetype)
    Vo = TensorKit.domain(t_mpo)[1]   # vR_o - domain-leg, must not use space(t,i)'s dual
    Vs = TensorKit.domain(t_mps)[1]   # vR_s
    Fr = _bond_fuser(storagetype, Vo, Vs)

    raw = _contract_physical(t_mpo, t_mps)
    step1 = _fuse_left(raw, Fl)
    result = _fuse_right(step1, Fr)
    return result, Fr
end

"""
    apply(mpo::MPOperator, mps::MPState) -> MPState

Apply `mpo` to `mps` via the default strategy, [`ExactApply`](@ref)`()`. See
[`ApplyStrategy`](@ref) for the alternative, [`CompressedApply`](@ref).
"""
apply(mpo::MPOperator, mps::MPState) = apply(mpo, mps, ExactApply())

"""
    apply(mpo::MPOperator, mps::MPState, ::ExactApply) -> MPState{UnknownGauge,S}

The exact, untruncated MPO-applied-to-MPS product (Schollwoeck's TEBD/DMRG building block): at
every bond, the new bond dimension is exactly `D_mpo * D_mps` (the bond-leg pair is fused via a
genuine `TensorKit.isomorphism`, not a lossy projection). Deliberately does **not** truncate - a
caller wanting a compressed result either uses [`CompressedApply`](@ref) or calls
[`canonicalize`](@ref) with `bond_cutoff` afterward, mirroring this codebase's existing separation
of construction ([`to_mps`](@ref)) from truncation, and MPSKit.jl's own `mpo*mps`/`changebonds!`
separation.

Per site, contracted via [`_apply_site`](@ref): (1) [`_contract_physical`](@ref) contracts `mpo`'s
bra leg against `mps`'s physical leg (the one leg that must match); (2) [`_fuse_left`](@ref)/
[`_fuse_right`](@ref) fuse the surviving virtual-leg pairs into single new bond legs via isomorphism.
The same fuser built as one site's right-bond fuser is reused as the next site's left-bond fuser -
this consistency is what keeps the resulting bond legs correctly identified across the chain.

Result is tagged [`UnknownGauge`](@ref) (no isometry structure is guaranteed by this construction).
"""
function apply(mpo::MPOperator{Gm,S}, mps::MPState{Gs,S}, ::ExactApply) where {Gm,Gs,S}
    L = length(mpo.sites)
    length(mps.sites) == L ||
        throw(ArgumentError("apply: mpo has $L site(s), mps has $(length(mps.sites))"))

    storagetype = TensorKit.storagetype(tensor(mps.sites[1]))
    Vo1 = TensorKit.space(tensor(mpo.sites[1]), 1)
    Vs1 = TensorKit.space(tensor(mps.sites[1]), 1)
    Fl = _bond_fuser(storagetype, Vo1, Vs1)

    new_sites = Vector{QProcess}(undef, L)
    for i in 1:L
        result, Fr = _apply_site(
            tensor(mpo.sites[i]), tensor(mps.sites[i]), Fl, storagetype
        )
        new_sites[i] = _finalize_site(Val(1), result)
        Fl = Fr
    end

    return MPState(new_sites, UnknownGauge(), 0, 0, nothing, hypot(mpo.ε, mps.ε))
end

"""
    apply(mpo::MPOperator, mps::MPState, strategy::CompressedApply) -> MPState{LeftCanonical,S}

Schollwoeck §5.3's fused procedure: requires `mpo`/`mps` already canonicalized (checked via
`is_canonical`, throwing a clear `ArgumentError` otherwise - callers `canonicalize` first if
needed, the same explicit-precondition style as [`to_vidal`](@ref)'s left-canonical-only
requirement). Sweeps left-to-right doing the same per-site contract-then-fuse
([`_apply_site`](@ref)) as [`ExactApply`](@ref), but immediately SVD-truncates each freshly-fused
bond (reusing [`advance_bond!`](@ref)'s existing SVD-cut machinery with `strategy.bond_cutoff`)
before folding the remainder into the next site's freshly-computed content - never materializing
the full untruncated bond at any point. `ε` accumulated via `QuadratureTruncationErrorAccumulator`,
matching [`canonicalize`](@ref)'s own error-accumulation convention.
"""
function apply(
    mpo::MPOperator{Gm,S}, mps::MPState{Gs,S}, strategy::CompressedApply
) where {Gm,Gs,S}
    is_canonical(mpo) || throw(
        ArgumentError(
            "apply(...,CompressedApply(...)) requires mpo already canonicalized - call " *
            "canonicalize(mpo, ...) first.",
        ),
    )
    is_canonical(mps) || throw(
        ArgumentError(
            "apply(...,CompressedApply(...)) requires mps already canonicalized - call " *
            "canonicalize(mps, ...) first.",
        ),
    )
    L = length(mpo.sites)
    length(mps.sites) == L ||
        throw(ArgumentError("apply: mpo has $L site(s), mps has $(length(mps.sites))"))

    storagetype = TensorKit.storagetype(tensor(mps.sites[1]))
    Vo1 = TensorKit.space(tensor(mpo.sites[1]), 1)
    Vs1 = TensorKit.space(tensor(mps.sites[1]), 1)
    Fl = _bond_fuser(storagetype, Vo1, Vs1)

    accumulator = QuadratureTruncationErrorAccumulator()
    sites = Vector{QProcess}(undef, L)
    T, Fr = _apply_site(tensor(mpo.sites[1]), tensor(mps.sites[1]), Fl, storagetype)

    for i in 1:(L - 1)
        site_tensor, remainder = advance_bond!(
            LeftRight(), Val(1), T, i; bond_cutoff=strategy.bond_cutoff, accumulator
        )
        sites[i] = _finalize_site(Val(1), site_tensor)

        T_next, Fr_next = _apply_site(
            tensor(mpo.sites[i + 1]), tensor(mps.sites[i + 1]), Fr, storagetype
        )
        Am = _regroup_first_out(T_next)
        T = _regroup_bulk_site(remainder * Am)
        Fr = Fr_next
    end
    sites[L] = _finalize_site(Val(1), T)

    ε = hypot(mpo.ε, mps.ε, something(finalize!(accumulator), 0.0))
    return MPState(sites, LeftCanonical(), L, L + 1, L, ε)
end

# SECTION -  apply_gate - local 1-/2-site gate application at a driver-supplied site range

# Regularized pseudo-inverse of a diagonal λ: 1/x for entries above tol, 0 otherwise - the
# standard iTEBD/Vidal-form convention for stripping a possibly-degenerate Schmidt weight back
# off a Γ tensor. A plain `inv` blows up (Inf) on an exactly-zero singular value (common right off
# a product-state `to_vidal`), and `Inf` multiplied against whatever component of U/Vd isn't
# *exactly* numerically zero in that same channel produces NaN that then contaminates every later
# step. Zeroing the corresponding inverse-weight component instead is safe: a near-zero Schmidt
# value means that channel carries no physical amplitude, so whatever it would be divided into is
# itself physically irrelevant.
function _safe_inv_diag(λ::TensorKit.DiagonalTensorMap; tol::Real=1e-12)
    data = map(x -> x > tol ? inv(x) : zero(x), λ.data)
    return TensorKit.DiagonalTensorMap(data, TensorKit.domain(λ)[1])
end
# A boundary λ is a plain identity isomorphism (not a DiagonalTensorMap) - always well-conditioned,
# so a plain `inv` is exact and safe here.
_safe_inv_diag(λ; tol::Real=1e-12) = inv(λ)

# 1-site: contract gate's (σ_ket)←(σ_bra) tensor into the site's physical leg directly. No SVD,
# no bond-dimension change.
function _apply_1site_gate(t, gate_tensor)
    t_iso = TensorKit.permute(t, ((2,), (1, 3)))          # σ ← (vL,vR)
    contracted = gate_tensor * t_iso                       # σ_ket ← (vL,vR)
    return TensorKit.permute(contracted, ((2, 1), (3,)))   # (vL,σ_ket) ← vR
end

"""
    apply_gate(state::MPState{MixedCanonical,S}, gate::QProcess, site_range::UnitRange{Int};
               bond_cutoff::Union{Int,Nothing}=nothing,
               accumulator::AbstractErrorAccumulator=NoOpErrorAccumulator()) -> MPState{MixedCanonical,S}

Apply a small local `gate` (1- or 2-site) to `state` at `site_range` - the missing primitive
between `trotterize`'s gate sequence and an actual TEBD sweep. Unlike [`apply`](@ref)
(whole-chain MPO×MPS), this touches only `site_range`.

**Precondition, not auto-fix**: `state.orthogonality_center` must already be at `site_range` (for
a 1-site gate) or adjacent to it (`first(site_range)` or `last(site_range)`, for a 2-site gate),
throwing `ArgumentError` otherwise - mirrors `apply(...,CompressedApply(...))`'s own precondition
style. `site_range` is a parameter the caller (a TEBD driver) supplies explicitly; this function
never infers position. Walking the center there is the driver's job (`canonicalize`), done once
per gate rather than repeated inside every call - keeps this primitive `O(1)`/`O(D³)` per call
instead of silently paying an `O(L)` resweep every time.

For a 1-site gate, `accumulator` is untouched (no truncation possible). For a 2-site gate, the two
sites are merged, the gate contracted into both physical legs, and the result re-split via
[`advance_bond!`](@ref) with `bond_cutoff` - truncation error flows into `accumulator` via
`record!`, exactly like [`apply`](@ref)'s `CompressedApply` strategy.
"""
function apply_gate(
    state::MPState{MixedCanonical,S},
    gate::QProcess,
    site_range::UnitRange{Int};
    bond_cutoff::Union{Int,Nothing}=nothing,
    accumulator::AbstractErrorAccumulator=NoOpErrorAccumulator(),
) where {S}
    n = length(site_range)
    if n == 1
        i = first(site_range)
        state.orthogonality_center == i || throw(
            ArgumentError(
                "apply_gate: orthogonality_center must be at site $i for a 1-site gate there, " *
                "got center=$(state.orthogonality_center) - canonicalize(state, MixedCanonicalize($i)) first.",
            ),
        )
        sites = copy(state.sites)
        sites[i] = _finalize_site(
            Val(1), _apply_1site_gate(tensor(state.sites[i]), tensor(gate))
        )
        return TensorTrain{MixedCanonical,S,1}(
            sites, state.llim, state.rlim, state.orthogonality_center, state.ε
        )
    elseif n == 2
        i, j = first(site_range), last(site_range)
        j == i + 1 || throw(
            ArgumentError(
                "apply_gate: site_range must span 1 or 2 contiguous sites, got $site_range",
            ),
        )
        state.orthogonality_center in (i, j) || throw(
            ArgumentError(
                "apply_gate: orthogonality_center must be at $i or $j for a 2-site gate there, " *
                "got center=$(state.orthogonality_center) - canonicalize(state, MixedCanonicalize($i)) first.",
            ),
        )

        Am = _regroup_first_out(tensor(state.sites[j]))       # (vL_j) ← (σ_j,vR_j)
        merged = _regroup_bulk_site(tensor(state.sites[i]) * Am)  # (vL_i,σ_i) ← (σ_j,vR_j)
        merged_iso = TensorKit.permute(merged, ((2, 3), (1, 4)))  # (σ_i,σ_j) ← (vL_i,vR_j)
        gated = tensor(gate) * merged_iso                          # (σ_i_ket,σ_j_ket) ← (vL_i,vR_j)
        gated_merged = TensorKit.permute(gated, ((3, 1), (2, 4)))  # (vL_i,σ_i_ket) ← (σ_j_ket,vR_j)

        # A single SVD of gated_merged serves both center-placement cases - only which factor
        # absorbs S (becoming the new center) differs: advance_bond!'s Val(P) machinery doesn't
        # fit here (it assumes exactly one *new* physical leg being cut off a growing block, not
        # two already-separate sites' physical legs merged together), so this calls
        # factorize_tensor directly, mirroring what advance_bond! itself does internally.
        U, Sv, Vd, ε = factorize_tensor(
            gated_merged, HasEntanglementSpectrum(); bond_cutoff
        )
        record!(accumulator, (; direction=LeftRight(), bond=i, ε))

        if state.orthogonality_center == i
            new_i = U                        # left-isometric
            new_j = _regroup_bulk_site(Sv * Vd)  # absorbs S, becomes the new center
            new_center = j
        else
            new_i = U * Sv                   # absorbs S, becomes the new center
            new_j = _regroup_bulk_site(Vd)    # right-isometric
            new_center = i
        end

        sites = copy(state.sites)
        sites[i] = _finalize_site(Val(1), new_i)
        sites[j] = _finalize_site(Val(1), new_j)
        return TensorTrain{MixedCanonical,S,1}(
            sites, new_center - 1, new_center + 1, new_center, state.ε
        )
    else
        throw(
            ArgumentError(
                "apply_gate: site_range must span 1 or 2 contiguous sites, got $site_range"
            ),
        )
    end
end

# Isolate leg 1 alone as codomain, attach λ, then restore t's *original* (codomain,domain) split.
function _absorb_lambda_left(t, λ)
    n = TensorKit.numind(t)
    c = TensorKit.numout(t)
    iso = TensorKit.permute(t, ((1,), Tuple(2:n)))
    attached = λ * iso
    return TensorKit.permute(attached, (Tuple(1:c), Tuple((c + 1):n)))
end

# Isolate the last leg alone as domain, attach λ, then restore t's original (codomain,domain) split.
function _absorb_lambda_right(t, λ)
    n = TensorKit.numind(t)
    c = TensorKit.numout(t)
    iso = TensorKit.permute(t, (Tuple(1:(n - 1)), (n,)))
    attached = iso * λ
    return TensorKit.permute(attached, (Tuple(1:c), Tuple((c + 1):n)))
end

# Isolate the last leg alone as domain and attach λ - unlike `_absorb_lambda_right`, does NOT
# restore t's original split, since the isolated split (n-1 codomain legs, 1 domain leg) is
# itself the target bulk-site shape (vL,σ)←(vR) - used to strip λ_right off a freshly-split `Vd`
# (originally 1 codomain leg, 2 domain legs) into a genuine bulk `Γ` site tensor.
function _absorb_lambda_right_to_bulk_shape(t, λ)
    n = TensorKit.numind(t)
    iso = TensorKit.permute(t, (Tuple(1:(n - 1)), (n,)))
    return iso * λ
end

"""
    apply_gate(state::MPState{VidalGauge,S}, gate::QProcess, site_range::UnitRange{Int};
               bond_cutoff::Union{Int,Nothing}=nothing,
               accumulator::AbstractErrorAccumulator=NoOpErrorAccumulator()) -> MPState{VidalGauge,S}

The classical Vidal (2003)/iTEBD local-update: applies `gate` at `site_range` without walking any
orthogonality center - every bond already carries its own Schmidt weights (`state.λs`), so the
gate contracts directly. Requires `state.λs !== nothing`.

For a 2-site gate at `i:(i+1)`, the standard local update: absorb the neighboring `λ`s (or an
identity at a chain boundary) into the merged 2-site block, contract the gate, re-split via
[`factorize_tensor`](@ref) with `bond_cutoff` (truncation error into `accumulator`, same convention
as the [`MixedCanonical`](@ref) method), renormalize the new bond's singular values (`Σλ²=1`), then
strip the *outer* λs back off (`inv(λ)`, the same `Γ = AL_tilde * inv(λ)` pattern [`to_vidal`](@ref)
already uses) to recover genuine `Γ` tensors.
"""
function apply_gate(
    state::MPState{VidalGauge,S},
    gate::QProcess,
    site_range::UnitRange{Int};
    bond_cutoff::Union{Int,Nothing}=nothing,
    accumulator::AbstractErrorAccumulator=NoOpErrorAccumulator(),
) where {S}
    state.λs === nothing && throw(
        ArgumentError(
            "apply_gate(::MPState{VidalGauge}, ...) requires a populated λs field - call " *
            "to_vidal on a LeftCanonical chain first.",
        ),
    )
    L = length(state.sites)
    n = length(site_range)
    if n == 1
        i = first(site_range)
        sites = copy(state.sites)
        sites[i] = _finalize_site(
            Val(1), _apply_1site_gate(tensor(state.sites[i]), tensor(gate))
        )
        return MPState{VidalGauge,S}(sites, 0, 0, nothing, state.ε, state.λs)
    elseif n == 2
        i, j = first(site_range), last(site_range)
        j == i + 1 || throw(
            ArgumentError(
                "apply_gate: site_range must span 1 or 2 contiguous sites, got $site_range",
            ),
        )

        Vl = TensorKit.space(tensor(state.sites[i]), 1)
        Vr = TensorKit.space(tensor(state.sites[j]), 3)'
        st = TensorKit.storagetype(tensor(state.sites[i]))
        λ_left = i == 1 ? TensorKit.isomorphism(st, Vl, Vl) : state.λs[i - 1]
        λ_right = j == L ? TensorKit.isomorphism(st, Vr, Vr) : state.λs[j]
        λ_mid = state.λs[i]

        Am = _regroup_first_out(tensor(state.sites[j]))
        merged = _regroup_bulk_site((tensor(state.sites[i]) * λ_mid) * Am)
        merged = _absorb_lambda_left(merged, λ_left)
        Θ = _absorb_lambda_right(merged, λ_right)

        merged_iso = TensorKit.permute(Θ, ((2, 3), (1, 4)))
        gated = tensor(gate) * merged_iso
        gated_merged = TensorKit.permute(gated, ((3, 1), (2, 4)))

        # A low-entanglement Vidal chain (e.g. straight off a product-state to_vidal, where every
        # bond's λ is exactly rank-1) can make gated_merged an *exactly* rank-deficient matrix -
        # not just numerically small singular values, but genuinely zero ones (a Néel-state XXZ
        # quench: every bond starts exactly rank-1, so even the very first gate hits this). Both
        # LAPACK `gesdd` (`SafeDivideAndConquer`, the default) and `gesvd` (`QRIteration`) proved
        # unreliable for this exact-degeneracy case on this platform (macOS Accelerate) -
        # `gesdd` crashes outright (`ArgumentError: invalid argument #4`), `gesvd` silently returns
        # `NaN` singular values or fails to converge (`LAPACKException(3)`) a few gates later once
        # the degeneracy compounds. `LAPACK_Jacobi` (one-sided Jacobi SVD) handles this exact input
        # correctly (confirmed via a minimal single-nonzero-entry reproduction) - Jacobi SVD is the
        # standard robust choice for exactly-degenerate/rank-deficient inputs, at some cost relative
        # to `gesdd`/`gesvd` (fine here - always small local 2-site blocks). `trunc_tol` additionally
        # *drops* any negligible singular value from the kept bond dimension outright (Schollwöck
        # 2011's standard treatment) rather than merely protecting the later `_safe_inv_diag`/
        # renormalization steps against dividing by it.
        U, Sv, Vd, ε = factorize_tensor(
            gated_merged,
            HasEntanglementSpectrum();
            bond_cutoff,
            trunc_tol=1e-10,
            alg=TensorKit.Factorizations.MatrixAlgebraKit.Jacobi(;
                driver=TensorKit.Factorizations.MatrixAlgebraKit.LAPACK()
            ),
        )
        record!(accumulator, (; direction=LeftRight(), bond=i, ε))

        # Guard against a genuinely (near-)zero-weight bond (a degenerate gate outcome, not
        # expected in practice once trunc_tol above drops negligible weights) rather than dividing
        # by ~0.
        Sv_norm = sqrt(real(TensorKit.tr(Sv' * Sv)))
        λ_new = Sv_norm > 1e-12 ? Sv / Sv_norm : Sv

        Γi = _absorb_lambda_left(U, _safe_inv_diag(λ_left))
        Γj = _absorb_lambda_right_to_bulk_shape(Vd, _safe_inv_diag(λ_right))

        λs = copy(state.λs)
        λs[i] = λ_new
        sites = copy(state.sites)
        sites[i] = _finalize_site(Val(1), Γi)
        sites[j] = _finalize_site(Val(1), Γj)
        return MPState{VidalGauge,S}(sites, 0, 0, nothing, state.ε, λs)
    else
        throw(
            ArgumentError(
                "apply_gate: site_range must span 1 or 2 contiguous sites, got $site_range"
            ),
        )
    end
end

# SECTION -  overlap - boundary/transfer-matrix sweep, never materializing the full state

"""
    overlap(bra::MPState, ket::MPState) -> Scalar

`⟨bra|ket⟩`, computed via Schollwoeck's standard boundary/transfer-matrix sweep: only a small
`D_bra × D_ket` boundary tensor is carried left-to-right, never a full-size intermediate object -
cost is `O(L · D_bra · D_ket · d · (D_bra+D_ket))`, not the `O(d^L)` a literal
`apply`-then-contract would imply.

Throws `DimensionMismatch` if `bra`/`ket` have different lengths or mismatched physical-leg spaces
at any site (checked implicitly by the per-site contraction failing to typecheck).
"""
function overlap(bra::MPState, ket::MPState)
    L = length(bra.sites)
    length(ket.sites) == L || throw(
        DimensionMismatch("overlap: bra has $L site(s), ket has $(length(ket.sites))")
    )

    storagetype = TensorKit.storagetype(tensor(ket.sites[1]))
    Vb1 = TensorKit.space(tensor(bra.sites[1]), 1)
    Vk1 = TensorKit.space(tensor(ket.sites[1]), 1)
    E = TensorKit.isomorphism(storagetype, Vb1, Vk1)   # vR_b_prev ← vR_k_prev, seeded at site1's left bond

    for i in 1:L
        t_bra = tensor(bra.sites[i])
        t_ket = tensor(ket.sites[i])

        ket_iso = TensorKit.permute(t_ket, ((1,), (2, 3)))   # vL_k ← (σ,vR_k)
        absorbed = E * ket_iso                                # vR_b_prev ← (σ,vR_k)
        absorbed2 = TensorKit.permute(absorbed, ((1, 2), (3,)))   # (vR_b_prev,σ) ← vR_k

        E = TensorKit.adjoint(t_bra) * absorbed2   # vR_b ← vR_k
    end

    return Scalar(TensorKit.removeunit(TensorKit.removeunit(E, 1), 1))
end

# SECTION -  local_expectation_value - cheap local/multi-site observable evaluation

"""
    local_expectation_value(state::MPState, ops::Vector{Pair{Int,QProcess}}) -> Scalar

Generalizes [`overlap`](@ref)'s own boundary/transfer-matrix sweep: at each site, contract the
physical leg against `dagger(state)`'s physical leg **through** the named operator when that site
has an entry in `ops` (reusing [`apply_gate`](@ref)'s `_apply_1site_gate` helper), or as a plain
identity pass-through otherwise. Cost `O(L·D²·d)`, same order as `overlap`, independent of how many
operators are in `ops` - no `to_mpo`/bond-dimension-`D_W` cost at all. `ops`' sites need not be
sorted or contiguous.
"""
function local_expectation_value(state::MPState, ops::Vector{<:Pair{Int,<:QProcess}})
    op_by_site = Dict(ops)
    L = length(state.sites)
    storagetype = TensorKit.storagetype(tensor(state.sites[1]))
    V1 = TensorKit.space(tensor(state.sites[1]), 1)
    E = TensorKit.isomorphism(storagetype, V1, V1)

    for i in 1:L
        t = tensor(state.sites[i])
        t_ket = haskey(op_by_site, i) ? _apply_1site_gate(t, tensor(op_by_site[i])) : t

        ket_iso = TensorKit.permute(t_ket, ((1,), (2, 3)))       # vL_k ← (σ,vR_k)
        absorbed = E * ket_iso                                    # vR_prev ← (σ,vR_k)
        absorbed2 = TensorKit.permute(absorbed, ((1, 2), (3,)))   # (vR_prev,σ) ← vR_k

        E = TensorKit.adjoint(t) * absorbed2   # vR_bra ← vR_ket, bra always uses the ORIGINAL t
    end

    return Scalar(TensorKit.removeunit(TensorKit.removeunit(E, 1), 1))
end

# SECTION -  norm - a QProcess/Scalar-formalism primitive, deliberately NOT LinearAlgebra.norm

# `norm` is intentionally *not* `export`ed from Subroutines (nor re-exported from Qritical):
# LinearAlgebra.norm is already brought into scope everywhere this package's tests run
# (`using LinearAlgebra`), and a *different*, non-extending `norm` generic function exported
# alongside it would collide the bare name the same way this codebase's own `dim`/`space` already
# do when both TensorKit's and Qritical's exports are in scope simultaneously. Reachable via
# explicit qualification (`Qritical.Subroutines.norm`) or `import Qritical.Subroutines: norm`.

# Wrap a plain number as a genuine (0,0)-legged Scalar, matching qprocess.jl's own
# `TensorMap(fill(x), one(V), one(V))` construction pattern for a bare number.
function _wrap_scalar(x::Number, V::TensorKit.ElementarySpace)
    triv = one(V)
    return Scalar(TensorKit.TensorMap(fill(ComplexF64(x)), triv, triv))
end

# ⟨center|center⟩ as a genuine QProcess composition: treat the center site's tensor with *all*
# its legs as codomain (a State over (vL,σ,vR) jointly), then adjoint(that) ∘ that reduces to a
# Scalar directly - mathematically ‖center‖², reached through the same categorical composition
# `overlap`/`∘` already use rather than an ad hoc TensorKit.norm call.
function _center_norm_squared(center::QProcess)
    t = tensor(center)
    full = TensorKit.permute(t, (Tuple(1:TensorKit.numind(t)), ()))
    s = State(full)
    return adjoint(s) ∘ s
end

# O(1) shortcut for a chain with a known orthogonality center (LeftCanonical/RightCanonical/
# MixedCanonical): the surrounding isometric sites contract to identity by construction, so only
# the center site needs touching - mirrors MPSKit's own
# `TensorKit.norm(ψ::FiniteMPS) = norm(ψ.AC[Int(ψ.center)])`.
function _norm_from_center(state::MPState)
    center = state.sites[state.orthogonality_center]
    sq = _center_norm_squared(center)
    return _wrap_scalar(sqrt(real(value(sq))), TensorKit.space(tensor(center), 1))
end

# General fallback (no known center): the full boundary sweep.
function _norm_from_overlap(state::MPState)
    sq = overlap(state, state)
    return _wrap_scalar(sqrt(real(value(sq))), TensorKit.space(tensor(state.sites[1]), 1))
end

"""
    norm(state::MPState) -> Scalar

`‖state‖` (not `‖state‖²`), gauge-dispatched: [`LeftCanonical`](@ref)/[`RightCanonical`](@ref)/
[`MixedCanonical`](@ref) chains (known `orthogonality_center`) use an O(1) shortcut touching only
the center site; [`VidalGauge`](@ref)/[`UnknownGauge`](@ref) chains fall back to the general
[`overlap`](@ref)-based boundary sweep. Returns a [`Scalar`](@ref), not a plain `Real` - see the
module note above for why this deliberately does not extend `LinearAlgebra.norm`.
"""
norm(state::MPState{LeftCanonical,S}) where {S} = _norm_from_center(state)
norm(state::MPState{RightCanonical,S}) where {S} = _norm_from_center(state)
norm(state::MPState{MixedCanonical,S}) where {S} = _norm_from_center(state)

# O(1) shortcut for a genuine Vidal-gauged chain: ‖ψ‖² = Σₐ λ[i][a]² for any internal bond `i`
# (all bonds agree by construction) - cheaper even than `_norm_from_center`, since it never
# touches a Γ tensor at all, just reads one λ array. Falls back to the general boundary sweep
# when `λs` is unpopulated (e.g. a bare fixture chain built with no real λ data).
function norm(state::MPState{VidalGauge,S}) where {S}
    state.λs === nothing && return _norm_from_overlap(state)
    λ = state.λs[1]
    sq = real(TensorKit.tr(λ' * λ))
    return _wrap_scalar(sqrt(sq), TensorKit.space(tensor(state.sites[1]), 1))
end
norm(state::MPState{UnknownGauge,S}) where {S} = _norm_from_overlap(state)
