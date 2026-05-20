# ADR 0005: SVD Truncation Tracking

## Status

Accepted (implementation in PLAN 5)

## Why this exists

The whole point of MPS is to live with approximations. When you compress a
tensor by keeping only the top r singular values and throwing away the rest,
you are making an explicit trade-off: smaller bond dimension in exchange for
some error. The current `factorize_with_svd` makes this trade silently — you
pass a threshold, it discards values, you get back U, Σ, Vt and nothing else.
You have no idea how bad the approximation was.

This becomes a problem as soon as you do more than one SVD in sequence — which
is exactly what happens in TEBD (one SVD per bond per time step) or in the SVD
sweep that produces a canonical MPS. Schöllwöck (2011) §4.1.3 shows that the
total error after L−1 truncations is bounded by:

    ‖|ψ⟩ − |ψ_trunc⟩‖₂² ≤ 2 Σᵢ εᵢ(D)

where εᵢ(D) is the discarded weight at bond i. To check this bound in code, I
need the per-bond εᵢ values stored somewhere and a way to accumulate them. That
is the whole motivation for this ADR.

## The discarded weight

Schöllwöck defines εᵢ(D) as the sum of *squared* discarded singular values:

    εᵢ(D) = Σⱼ₌ᵣ₊₁ᴿ σⱼ²

This is confirmed by the Von Delft/LMU course notes (MPS-II.2, Eq. 20), which
derive the truncation error directly:

    ‖|ψ⟩ − |ψ̃⟩‖₂² = Σ_{λ=r'+1}^{r} sλ² = sum of squares of discarded singular values

So `discarded_weight` is also the squared 2-norm error on the state, when the
input is a bipartite state matrix. This is also what the MPS/DMRG community
calls the **discarded weight** — the trace of the reduced density matrix
eigenvalues that were thrown away: `Σ w_a_discarded = Σ σ²_a_discarded`.

I was initially confused about whether to store the squared error or its square
root. The reason to store the squared version is that Eq. (58) adds εᵢ
*linearly*. If I stored `√εᵢ` instead, I would need to square before summing,
which obscures the direct correspondence with the paper formula.

The Frobenius distance to the truncated tensor is `√εᵢ`, and that is what
MPSKit.jl returns as its scalar truncation error. Both are useful, so the
design exposes both: `discarded_weight` (squared, stored) and
`truncation_error(r)` (square root, derived function).

## Why max_rank is the primary truncation criterion

The Von Delft notes (MPS-II.2, Eq. 18 and surrounding text) state: "The
truncation strategy (18) minimizes the truncation error." Strategy (18) is
keeping the r' *largest* singular values. This is the Eckart-Young theorem —
the optimal rank-r' approximation of a matrix in Frobenius norm is obtained by
keeping the top r' singular values. This justifies `max_rank` as the primary
criterion in `TruncationSpec`: it is not just a budget constraint, it is
literally the optimal strategy for a given bond dimension.

## Convention comparison with established libraries

I looked at how ITensors.jl and MPSKit.jl handle this before deciding:

**ITensors.jl** returns a `TruncSVD` struct containing a `Spectrum`. The
`Spectrum.truncerr` field accumulates the discarded eigenvalues of the density
matrix — i.e., `Σ σ²_discarded`. This matches Schöllwöck's εᵢ and our
`discarded_weight` field. The default behaviour divides by the total norm
(relative cutoff), which we defer.

**MPSKit.jl / MatrixAlgebraKit.jl** returns a 4-tuple `(U, S, Vᴴ, ϵ)` where
`ϵ = √(Σ σ²_discarded)` — the 2-norm of the discarded singular values. This
matches our derived function `truncation_error(r)`.

So our `discarded_weight` field agrees with ITensors and Schöllwöck; our
`truncation_error` function agrees with MPSKit. The `kept_weight` accessor
(`frobenius_norm_sq - discarded_weight = ‖Ψ̃‖_F²`) has no direct counterpart
in either library but falls directly out of Schöllwöck §4.1.1 Eq. (27), which
notes that the singular values of the truncated state "must be rescaled if
normalization is desired."

## The TruncationSpec design

The truncation criteria live in a separate `TruncationSpec` value type rather
than as keyword arguments to `factorize_with_svd`. This matters because the
spec is a first-class object that can be passed around, stored, printed, and
reused across multiple SVD calls consistently. Keyword arguments can't do that.

Two criteria are supported:

- `max_rank::Union{Int, Nothing}` — keep the top D singular values. This is
  the natural language of MPS bond dimension control.
- `abs_threshold::Union{Float64, Nothing}` — discard singular values below ε.
  This was the only option in the previous `factorize_with_svd`.

Both can be active simultaneously: a singular value must pass both filters to
be kept. `TruncationSpec()` with no arguments means "keep everything" — a safe
default that leaves `factorize_with_svd` usable without any spec argument.

**Deferred: relative threshold** (`σ/σ_max < ε_rel`). ITensors uses this as
its default, and it makes sense for states where the singular value scale
varies wildly between bonds. But it adds a data dependency (need to see `σ_max`
before computing the mask) and is not needed for the TEBD exercises, which
parameterise truncation purely by bond dimension D.

## The TruncationLog design

Neither ITensors nor MPSKit accumulate errors across bonds automatically — the
caller is responsible. I want to make the Schöllwöck bound computable without
ceremony, so `TruncationLog` is a thin mutable wrapper around
`Vector{TruncationResult}` that the caller pushes results into bond by bond.

The key query is `total_discarded_weight(log) = Σ εᵢ`. Multiplying by 2 gives
the Schöllwöck worst-case bound. I deliberately do not bake the factor of 2
into the log: the factor of 2 is a property of the MPS error bound derivation,
not a property of the truncation primitive. Keeping them separate means the log
is also usable in other contexts where the accumulation formula differs.

## What is deferred

- **Relative threshold** — add to `TruncationSpec` when needed
- **Per-bond labelling in TruncationLog** — knowing *which* bond each
  `TruncationResult` came from is MPS-level bookkeeping; the log is agnostic
- **Display/show methods** — plain struct printing is fine for now
- **Integration with PLAN 3 Isometry types** — `factorize_with_svd` will
  eventually annotate U and Vt as `Isometry`; this plan only touches the
  truncation metadata

## References

- Schöllwöck, Ann. Phys. 326 (2011), §4.1.1, Eqs. (24–27) — Schmidt
  coefficients, reduced density matrix, von Neumann entropy, truncated state
- Schöllwöck, Ann. Phys. 326 (2011), §4.1.3, Eq. (58) — worst-case error
  bound for sequential MPS truncation
- Von Delft/LMU course notes, MPS-II.2, Eqs. (18–20) — optimal truncation
  strategy (Eckart-Young), truncation error = sum of squares of discarded
  singular values; singular value decay plot showing kept vs. discarded regions
- ITensors.jl `NDTensors/src/truncate.jl` — `Spectrum.truncerr` convention
- MatrixAlgebraKit.jl `src/interface/truncation.jl` — `TruncationByError` and
  `svd_trunc` return convention
- ADR 0004 (`0004-special-tensor-types.md`) — SVD return type; PLAN 3 will
  eventually wrap U and Vt as `Isometry`
- GitHub issue #63
