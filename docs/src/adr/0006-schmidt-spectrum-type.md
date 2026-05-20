# ADR 0006: SchmidtSpectrum Type

## Status

Accepted (implementation in PLAN 6)

## Why this exists

There is a conceptual gap in the current API. `get_schmidt_coefficients`
returns a plain `Vector{Float64}` and `get_entanglement_entropy` is a
standalone function that takes that vector. There is nothing in the types that
says "these are normalised Schmidt coefficients from a bipartite quantum state"
as opposed to, say, any other vector of floats between 0 and 1. The invariant
`Σ λᵢ² ≤ 1` is checked at runtime with an `@assert` in `get_entanglement_entropy`,
but it is not expressed in the type.

Beyond just type safety, there is a practical question: when you truncate the
Schmidt spectrum (keeping only the top r coefficients), you lose track of the
full Schmidt rank R. The returned vector has length r with no memory of R, so
you cannot tell after the fact how much truncation occurred. This becomes
relevant when comparing entanglement across bipartitions, or when you want to
know whether your bond dimension budget was actually saturated.

## Is "Schmidt spectrum" an apt name here?

This is worth being careful about, because SVD is used in two distinct ways in
MPS algorithms and the name only applies to one of them.

**The Schmidt spectrum IS defined at the orthogonality center of a mixed
canonical form.** When an MPS is in mixed canonical form with the orthogonality
center at bond i, all tensors to the left are left-isometric and all tensors to
the right are right-isometric. The singular values at bond i are then exactly
the Schmidt coefficients for the bipartition A = {1,...,i} and B = {i+1,...,L}:

    Ψ = Σₐ λₐ |aₐ⟩_A |aₐ⟩_B

with Σ λₐ² = 1 (for a normalised state). The Schmidt rank, entanglement entropy,
and the error bound on truncation all refer to this specific setup.

**During a canonicalisation sweep, the singular values are NOT Schmidt
coefficients in general.** When you sweep left-to-right making left-isometries,
each local SVD gives you singular values of the contracted local tensor, not of
the global state bipartitioned at that bond. The left environment is built up
incrementally and is not yet fully contracted; the singular values at
intermediate bonds reflect only the local structure, not the global entanglement.
Only at the final bond (or at the orthogonality center if you stop mid-sweep)
are the singular values globally meaningful in the Schmidt sense.

Vidal's Γ-Λ notation makes this explicit by storing the Schmidt coefficients
as diagonal Λ matrices explicitly on each bond. Left/right canonical forms fold
the Λ matrices into the site tensors Γ for computational efficiency, which is
why the Schmidt coefficients are not directly visible unless you stop at the
right place.

**Conclusion:** the name SchmidtSpectrum is apt for what `get_schmidt_coefficients`
computes — it processes a bipartite state matrix that has already been reshaped
for a specific bipartition, and normalises the singular values so that Σ λᵢ² ≤ 1.
The user is responsible for ensuring they call it in a context where that
normalisation is physically meaningful (i.e., the state is normalised and the
bipartition is the intended one). The type does not encode the bipartition or
verify that the input is a genuine quantum state; it encodes what came out of
the computation.

## Is a struct worth having over a plain Vector?

The argument for a plain `Vector{Float64}` is simplicity: Julia's base array
functions (`sort`, `sum`, `length`, broadcasting) all work on it out of the
box, and adding a wrapper type means adding methods.

The argument for a struct is that the wrapper carries information that the
plain vector cannot:

1. **Full rank R** — after truncation, the vector has length r, not R. Without
   storing R, you cannot reconstruct the truncation ratio r/R or the discarded
   weight `1 - Σ λᵢ²_kept` correctly without keeping separate bookkeeping.

2. **Semantic invariant** — `SchmidtSpectrum` signals to the reader (and the
   type system) that `Σ λᵢ² ≤ 1` is expected to hold, and that the values are
   normalised Schmidt coefficients, not raw singular values.

3. **Natural method home** — `entanglement_entropy` belongs on a
   `SchmidtSpectrum`, not as a standalone function that accepts any vector and
   checks normalisation with an assert at runtime.

The design is lightweight: two fields, several short methods. The overhead is
not a meaningful cost.

## The design

```julia
struct SchmidtSpectrum
    coefficients::Vector{Float64}   # λᵢ in descending order, Σ λᵢ² ≤ 1
    full_rank::Int                  # R before any truncation
end
```

Derived quantities as functions (not fields):

- `schmidt_rank(s)` = `length(s.coefficients)` — r, the kept rank
- `is_truncated(s)` = `schmidt_rank(s) < s.full_rank`
- `entanglement_entropy(s)` = `-Σ λᵢ² log₂ λᵢ²` — von Neumann entropy in bits
- `discarded_weight(s)` = `1.0 - sum(abs2, s.coefficients)` — the fraction
  of state norm lost to truncation; equals zero for an untruncated spectrum
  and equals Schöllwöck's εᵢ when the input state was normalised

`get_schmidt_coefficients` is refactored to return `SchmidtSpectrum` instead
of `Vector{Float64}`. The existing `get_entanglement_entropy(coeffs::Vector)`
function is replaced by `entanglement_entropy(s::SchmidtSpectrum)`.

## Relationship to TruncationResult (PLAN 5)

`TruncationResult` and `SchmidtSpectrum` track related but different things:

- `TruncationResult` — attached to a raw SVD call (`factorize_with_svd`);
  stores `discarded_weight = Σ σ²_discarded` and `frobenius_norm_sq = Σ σ²_all`
  in absolute (unnormalised) units. Does not assume the input is a quantum state.

- `SchmidtSpectrum` — attached to a bipartite quantum state; stores normalised
  coefficients. `discarded_weight` is `1 - Σ λᵢ²` and is dimensionless (a
  fraction of the state norm). Assumes the input was a normalised state.

`SchmidtSpectrum` does not contain a `TruncationResult` — the two are produced
by different functions at different levels of abstraction and should remain
independent. A caller who needs both can call `factorize_with_svd` and
`get_schmidt_coefficients` in sequence.

## Renormalization after truncation

Von Delft/LMU course notes (MPS-II.2, Eq. 19) make explicit that the
truncated state `|ψ̃⟩ = Σ_{λ=1}^{r'} |λ⟩_B |λ⟩_A sλ` is generally not
normalized, and must be rescaled:

    sλ  →  sλ · [Σ_{λ'=1}^{r'} sλ'²]^{-1/2}

In our notation, `SchmidtSpectrum` stores already-normalised λᵢ = sᵢ/‖Ψ‖.
After truncation, Σ λᵢ² = 1 - discarded_weight(s) < 1. To recover a
normalised state, the rescaling factor is `1 / sqrt(1 - discarded_weight(s))`.
This is exposed as `renormalize(s::SchmidtSpectrum) → SchmidtSpectrum`.

## Maximum entanglement entropy

Von Delft/LMU (Eq. 16): for a fixed Schmidt rank r, entanglement entropy is
maximised when all singular values are equal: `sλ = r^{-1/2}`, giving
`S_max = log₂(r)` bits. This is exposed as:

    max_entanglement_entropy(s::SchmidtSpectrum) = log2(schmidt_rank(s))

Comparing `entanglement_entropy(s)` with `max_entanglement_entropy(s)` gives a
dimensionless measure of how close the state is to maximally entangled at this
bipartition.

## Physical interpretation of the singular value spectrum

The Von Delft notes motivate truncation with a log-scale plot of sλ vs. λ:
kept singular values trace a smoothly decaying curve; discarded ones fluctuate
at the noise floor several orders of magnitude below. The truncation is "cheap"
(low error) when the spectrum decays rapidly — exponential decay on the log
plot, which is the hallmark of gapped 1D systems satisfying an area law.
Slow or algebraic decay signals a critical or highly entangled state where
large bond dimension is unavoidable.

`discarded_weight(s)` is the quantitative version of "how much of the spectrum
was below the noise floor" — small discarded weight confirms the rapid-decay
condition is met.

## What is deferred

- **Canonical form verification** — checking that a given `SchmidtSpectrum`
  was produced at the orthogonality center of a canonical form is out of scope;
  that is MPS-level semantics
- **Bipartition metadata** — storing which bipartition produced the spectrum
  (a `Bisection`) could be useful but adds coupling; deferred until MPS types
  exist to make the context clear
- **Rényi entropies** — `Sₙ = (1/(1-n)) log Σ λᵢ²ⁿ`; natural extension but
  not needed for Exercise 2 (canonical forms)

## References

- Schöllwöck, Ann. Phys. 326 (2011), §4.1.1, Eqs. (24–27) — Schmidt
  decomposition, eigenvalues of reduced density matrix, von Neumann entropy
- Vidal, Phys. Rev. Lett. 91 (2003) — Γ-Λ notation making Schmidt spectra
  explicit on each bond
- Von Delft/LMU course notes, MPS-II.2, Eqs. (15–19) — entanglement entropy,
  maximal entanglement, truncated state, renormalization after truncation,
  singular value decay plot
- ADR 0005 (`0005-svd-truncation-tracking.md`) — TruncationResult at the raw
  SVD level; SchmidtSpectrum sits above this
- GitHub issue #64
