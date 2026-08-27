# SECTION -  Sweep helpers on existing MPS 
#
# Unlike _left_sweep / _right_sweep (which start from a dense d^L tensor),
# these helpers operate directly on the L valence-3 site tensors of an existing
# MPS by propagating a "carry" factor between neighbours:
#
#   Left sweep at site i:  reshape A_i → (χL·d, χR), SVD, store U as the new
#   left-canonical A_i, absorb Σ·Vd into A_{i+1} via a matrix multiply.
#
#   Right sweep at site i: reshape B_i → (χL, d·χR), SVD, store Vd as the new
#   right-canonical B_i, absorb U·Σ into B_{i-1}.
#
#   Per-bond cost: O(χ²·d) — linear in L, not exponential in d.  This is the
#   fundamental efficiency of MPS methods.
#
# Each helper mutates the tensor/bond_svs vectors in place and returns the
# accumulated truncation error ε.
#
# Trade-off note (§3.2):
#   canonical_error (public API) tests only LEFT isometry (‖A†A − I‖).
#   Right-isometry checks (‖BB† − I‖) live in _right_isometry_error_mps,
#   used internally by is_canonical.  Extend canonical_error with a direction
#   kwarg if a public right-isometry API is needed later.

function _left_sweep_mps!(
    tensors::Vector{QTensor},
    bond_svs::Vector{SingValSpectrum},
    range::AbstractUnitRange{Int},
    trunc::AbstractTrunc,
)
    # The `!` at the end of the function name is a Julia CONVENTION (not syntax) meaning
    # "this function MUTATES its arguments in place." Python has no such convention,
    # but it's similar to how numpy functions with `out=` write to existing arrays.
    # Here `tensors` and `bond_svs` are modified directly — no new Vector is returned.
    #
    # `AbstractUnitRange{Int}` is the supertype for ranges like `1:5` or `3:(L-1)`.
    # In Python these would just be `range(1, 6)` objects.

    ε_total = 0.0
    for i in range
        # `range` here is the loop variable (a range like 1:4), not Python's built-in `range`.
        # In Julia, `for i in 1:(L-1)` is the standard; here `range` is passed as an argument.

        A = tensors[i].data                        # (χL, d, χR): current site tensor's raw array
        χL, d, χR = size(A)
        # `size(A)` returns a Tuple{Int,Int,Int}. Destructuring into χL, d, χR like Python unpacking.

        M = reshape(A, χL * d, χR)
        # Flatten the LEFT (virtual + physical) indices together, leaving the right virtual separate.
        # Physics: this is the "bipartition" of the chain at bond i|i+1.
        # In numpy: M = A.reshape(χL * d, χR)

        F = _robust_svd(M)
        # Full SVD: F.U (left unitary), F.S (singular values), F.Vt (right unitary conjugate-transposed).
        # Physics: this SVD gives us the Schmidt decomposition at bond i|i+1.
        # The singular values in F.S ARE the Schmidt values (up to noise).

        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        # Golub-Van Loan threshold: σ_min > n × ε_machine × σ_max.
        # This removes numerical noise (floating-point artifacts from the SVD algorithm)
        # before truncation, preventing ghost Schmidt values from inflating bond dimension.

        S_clean = filter(s -> s > tol, F.S)
        # Keep only singular values above the noise floor.
        # `filter(predicate, collection)` = Python's `[s for s in F.S if s > tol]`.

        r, ε_bond = _truncate_singular_values(S_clean, trunc)
        # Apply the user's truncation strategy. Returns:
        #   r = how many singular values to keep
        #   ε_bond = L2 norm of discarded values (= truncation error at this bond)

        svs = F.S[1:r]      # top-r singular values (Julia: 1-indexed, inclusive range 1..r)
        ε_total = hypot(ε_total, ε_bond)
        # Accumulate in QUADRATURE, not by plain addition. ε is a 2-norm, so ε² is the
        # weight discarded at this bond; weights add, norms do not. `hypot(a, b)` computes
        # sqrt(a² + b²) without overflowing on the intermediate squares.

        tensors[i] = QTensor(
            reshape(F.U[:, 1:r], χL, d, r), (upper(:vL, χL), upper(:σ, d), lower(:vR, r))
        )
        # Physics: U has orthonormal columns → A_i†A_i = I_r (left isometry).
        # Reshape U from (χL*d, r) to (χL, d, r) and wrap with index metadata.
        # The new right bond has dimension r (possibly truncated from χR) and is Lower-variance
        # (arrow points AWAY from the OC, which is to the right during a left sweep).
        # This IN-PLACE assignment replaces tensors[i] — the `!` convention signals this mutation.

        normalized = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        # Check if ||svs||² ≈ 1.0 (the bond is normalized).
        # `sum(abs2, svs)` = Σ sᵢ² — this is the norm squared of the singular value vector.
        # `sqrt(eps(...))` is a relaxed tolerance (≈ 1e-8 for Float64) because accumulated
        # floating-point error after L SVDs can be larger than ε_machine ≈ 1e-16.

        bond_svs[i + 1] = SingValSpectrum(svs, ε_bond, normalized)
        # Store the Schmidt spectrum for bond i|i+1. Bond index i+1 because Julia is 1-indexed
        # and bond_svs[1] is the LEFT boundary (trivial), so bond_svs[i+1] = bond after site i.

        # Absorb Σ·Vd into next tensor; tagged left-canonical (Up,Up,Low) — transient
        # if the sweep continues, final if site i+1 is the left form's norm carrier.
        # Mixed-form configs retag their centre site to all-Upper afterwards.
        carry = Diagonal(svs) * F.Vt[1:r, :]    # (r, χR): the "carry" factor propagating right
        # `Diagonal(svs)` = r×r diagonal matrix (like np.diag(svs)). Avoids a full matrix multiply.
        # `F.Vt[1:r, :]` = first r rows of V† (shape r×χR). Matrix product: (r×r)(r×χR) → (r×χR).

        A_next = tensors[i + 1].data                # (χR, d_next, χR_next): the NEXT site's data
        _, d_next, χR_next = size(A_next)
        # `_` is Julia's convention for "I don't need this value" (same as Python's _ in unpacking).
        # We don't need the first dim of A_next since we know it equals χR from the current bond.

        merged = reshape(carry * reshape(A_next, χR, d_next * χR_next), r, d_next, χR_next)
        # This is the key contraction: (carry) × (A_next reshaped as a matrix)
        # Step 1: reshape A_next from (χR, d_next, χR_next) to a matrix (χR, d_next*χR_next)
        # Step 2: carry (r×χR) times A_next_matrix (χR×d_next*χR_next) → (r×d_next*χR_next)
        # Step 3: reshape result back to (r, d_next, χR_next) for the next site tensor.
        # Physics: we are absorbing the gauge factor (Σ·V†) into the next site, making site i
        # left-canonical while the gauge weight flows rightward.

        tensors[i + 1] = QTensor(
            merged, (upper(:vL, r), upper(:σ, d_next), lower(:vR, χR_next))
        )
        # Tag the next site with provisional index metadata. The (Upper, Upper, Lower) pattern
        # is the left-canonical convention — but this tensor is transient; it will be SVD'd
        # in the next iteration of the loop (unless i+1 is the last site of the sweep).
    end
    return ε_total
    # In Julia, the last evaluated expression is also the implicit return. But `return` is explicit here
    # for clarity. Compare Python's `return ε_total`.
end

function _right_sweep_mps!(
    tensors::Vector{QTensor},
    bond_svs::Vector{SingValSpectrum},
    range::AbstractUnitRange{Int},
    trunc::AbstractTrunc,
)
    # Mirror of _left_sweep_mps! but proceeding RIGHT→LEFT.
    # Sites in `range` become right-canonical (BB†=I).
    ε_total = 0.0
    for i in reverse(range)
        # `reverse(range)` iterates the range backwards: e.g. `reverse(2:5)` → 5, 4, 3, 2.
        # In Python: `reversed(range(2, 6))`. Julia's `reverse` on a UnitRange returns a StepRange.

        B = tensors[i].data                        # (χL, d, χR): current site tensor's raw array
        χL, d, χR = size(B)
        M = reshape(B, χL, d * χR)
        # For right sweep: GROUP the physical+right virtual together, leaving left virtual separate.
        # Physics: this bipartitions the chain at bond (i-1)|i.
        # The SVD of M gives right-canonical V† (orthonormal rows → BB†=I).

        F = _robust_svd(M)
        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        S_clean = filter(s -> s > tol, F.S)
        r, ε_bond = _truncate_singular_values(S_clean, trunc)
        svs = F.S[1:r]
        ε_total = hypot(ε_total, ε_bond)   # quadrature, as in the left sweep: discarded WEIGHTS add

        tensors[i] = QTensor(
            reshape(F.Vt[1:r, :], r, d, χR), (lower(:vL, r), upper(:σ, d), upper(:vR, χR))
        )
        # Physics: V† has orthonormal rows → B_i·B_i† = I_r (right isometry).
        # Reshape F.Vt[1:r, :] from (r, d*χR) to (r, d, χR).
        # Right-canonical convention: vL is Lower (arrow points LEFT/away from OC), vR is Upper.
        # Note: `lower(:vL, r)` here has dimension r (truncated bond dim), not the original χL.

        normalized = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        bond_svs[i] = SingValSpectrum(svs, ε_bond, normalized)
        # Note: in the right sweep, bond_svs[i] is the bond to the LEFT of site i (between i-1 and i).
        # This is consistent: bond_svs[1] = left boundary, bond_svs[L+1] = right boundary.

        # Absorb U·Σ into previous tensor; tagged right-canonical (Low,Up,Up) — transient
        # if the sweep continues, final if site i-1 is the right form's norm carrier.
        # Mixed-form configs retag their centre site to all-Upper afterwards.
        carry = F.U[:, 1:r] * Diagonal(svs)   # (χL, r): the carry propagating LEFTWARD
        # `F.U[:, 1:r]` = first r columns of U (shape χL×r).
        # `F.U[:, 1:r] * Diagonal(svs)` = (χL×r)(r×r) → (χL×r): absorbs singular values into U.
        # Physics: U·Σ is the non-isometric part that flows leftward to the previous site.

        A_prev = tensors[i - 1].data              # (χL_p, d_p, χL): the PREVIOUS site's data
        χL_p, d_p, _ = size(A_prev)
        # `χL` in `A_prev` is the shared bond dimension — the right virtual of site i-1
        # equals the left virtual of site i.

        merged = reshape(reshape(A_prev, χL_p * d_p, χL) * carry, χL_p, d_p, r)
        # Step 1: reshape A_prev from (χL_p, d_p, χL) to matrix (χL_p*d_p, χL)
        # Step 2: matrix multiply (χL_p*d_p, χL) × (χL, r) → (χL_p*d_p, r)
        # Step 3: reshape back to (χL_p, d_p, r)
        # Physics: the carry (U·Σ) is absorbed into the previous site, making site i
        # right-canonical while the gauge weight flows leftward.

        tensors[i - 1] = QTensor(merged, (lower(:vL, χL_p), upper(:σ, d_p), upper(:vR, r)))
        # Tag the previous site with provisional right-canonical index convention.
        # `upper(:vR, r)` — the right bond dimension is now r (truncated).
    end
    return ε_total
end

# Retag site k as the orthogonality centre: both bond arrows point INTO the
# centre, so every leg is Upper (domain). The data is untouched — with the
# trivial metric this is a pure re-labelling (a double bond-end flip).
function _tag_centre!(tensors::Vector{QTensor}, k::Int)
    # Physics: at the orthogonality centre, ALL bond arrows point INWARD.
    # This means both vL and vR are Upper (contravariant) — the convention
    # is that Upper = "arrow points in toward this tensor".
    # We only need to change the INDEX METADATA (variance tags), not the data array itself.
    χL, d, χR = size(tensors[k].data)
    tensors[k] = QTensor(tensors[k].data, (upper(:vL, χL), upper(:σ, d), upper(:vR, χR)))
    # Three `upper(...)` instead of the usual left-canonical (Upper, Upper, Lower) pattern.
    # `return nothing` = Python's `return None` — explicit that this function has no meaningful return.
    return nothing
end

# SECTION -  Canonicalize configs 

"""
    CanonicalizeConfig

Supertype for canonicalization configurations passed to [`canonicalize`](@ref).
Concrete subtypes: [`LeftCanonical`](@ref), [`RightCanonical`](@ref),
[`BondCanonical`](@ref), [`SiteCanonical`](@ref).
"""
# Another abstract type used purely for dispatch. `canonicalize(mps, config)` will
# call different methods depending on whether config is LeftCanonical, RightCanonical, etc.
# This is Julia's MULTIPLE DISPATCH — the function selected depends on ALL argument types,
# not just the first one (unlike Python's single dispatch / method resolution).
abstract type CanonicalizeConfig end

"""
    LeftCanonical(trunc = NoTrunc())

Config: sweep left-to-right, producing `CanonicalForm(L, L+1)`.
"""
struct LeftCanonical <: CanonicalizeConfig
    trunc::AbstractTrunc   # truncation strategy to apply at each bond during the sweep
end
# Default constructor: if no truncation argument given, use NoTrunc().
# In Python this would be: `@dataclass class LeftCanonical: trunc: AbstractTrunc = field(default_factory=NoTrunc)`
# Julia doesn't have default field values in structs; instead we define an extra constructor:
LeftCanonical() = LeftCanonical(NoTrunc())
# This is a OUTER CONSTRUCTOR — defined outside the struct, adding a new way to call it.
# When you call `LeftCanonical()`, Julia uses this method; `LeftCanonical(trunc)` uses the auto-generated one.

"""
    RightCanonical(trunc = NoTrunc())

Config: sweep right-to-left, producing `CanonicalForm(0, 1)`.
"""
struct RightCanonical <: CanonicalizeConfig
    trunc::AbstractTrunc
end
RightCanonical() = RightCanonical(NoTrunc())

"""
    BondCanonical(k, trunc = NoTrunc())

Config: mixed canonical with orthogonality centre at bond ``k \\leftrightarrow k+1``.
Sites ``1 \\ldots k-1`` become left-canonical, sites ``k+1 \\ldots L`` become
right-canonical, site ``k`` holds the full gauge weight.
Result is tagged `CanonicalForm(k, k+1)`.
"""
struct BondCanonical <: CanonicalizeConfig
    k::Int              # the site index that will hold the orthogonality centre (the "centre site")
    trunc::AbstractTrunc
end
# Default constructor with just k: truncation defaults to NoTrunc().
BondCanonical(k::Int) = BondCanonical(k, NoTrunc())
# Physics: "bond canonical at bond k" means:
#   Sites 1..k-1: left-canonical (A†A=I)
#   Site k: holds the singular values (the gauge weight)
#   Sites k+1..L: right-canonical (BB†=I)
# This is also called "mixed canonical form" in much of the MPS literature.

"""
    SiteCanonical(k, trunc = NoTrunc())

Config: orthogonality centre at site ``k``.
Sites ``1 \\ldots k-1`` left-canonical, sites ``k+1 \\ldots L`` right-canonical,
site ``k`` is the un-gauged centre tensor.
Result is tagged `CanonicalForm(k-1, k+1)` (centre excluded from both isometry ranges).
"""
struct SiteCanonical <: CanonicalizeConfig
    k::Int              # site index for the orthogonality centre
    trunc::AbstractTrunc
end
SiteCanonical(k::Int) = SiteCanonical(k, NoTrunc())
# Physics: similar to BondCanonical but site k is the un-gauged centre tensor (not a diagonal SV matrix).
# Difference from BondCanonical: site k carries the full non-isometric weight as a rank-3 tensor.

# SECTION -  canonicalize 

"""
    canonicalize(mps::FiniteMPS, config::CanonicalizeConfig) -> FiniteMPS

Re-gauge `mps` according to `config` by sweeping with SVD steps on the
existing site tensors.  The input MPS is not mutated; a new `FiniteMPS` is
returned.

Supported configs:

  - [`LeftCanonical`](@ref): full left sweep → `CanonicalForm(L, L+1)`
  - [`RightCanonical`](@ref): full right sweep → `CanonicalForm(0, 1)`
  - [`BondCanonical`](@ref): mixed form, centre at bond k → `CanonicalForm(k, k+1)`
  - [`SiteCanonical`](@ref): mixed form, centre at site k → `CanonicalForm(k-1, k+1)`

# Algorithm: carry propagation on an existing MPS

Unlike [`to_mps`](@ref), which builds an MPS from a dense ``d^L`` state tensor,
`canonicalize` operates directly on the ``L`` rank-3 site tensors.  At each
bond the sweep SVD-factorises the current site tensor and contracts the
"carry" factor into the neighbouring site:

  - **Left sweep** at site ``i``: reshape ``A_i`` to ``(\\chi_L d \\times \\chi_R)``,
    SVD → ``U \\Sigma V^\\dagger``.  Store ``U`` as the new (left-canonical) ``A_i``
    and absorb ``\\Sigma V^\\dagger`` into ``A_{i+1}`` via a matrix multiply.

  - **Right sweep** at site ``i``: reshape ``B_i`` to ``(\\chi_L \\times d \\chi_R)``,
    SVD → ``U \\Sigma V^\\dagger``.  Store ``V^\\dagger`` as the new (right-canonical)
    ``B_i`` and absorb ``U \\Sigma`` into ``B_{i-1}``.

Because the carry is always a ``(r \\times \\chi)`` matrix multiply, the per-bond
cost is ``O(\\chi^2 d)`` rather than ``O(d^L)``.  This is the fundamental
efficiency of MPS methods: re-gauging the whole chain costs linear time in ``L``.
"""
function canonicalize(mps::FiniteMPS, config::LeftCanonical)
    # Julia MULTIPLE DISPATCH: this specific method is called when config is LeftCanonical.
    # There are 4 versions of `canonicalize` — Julia picks the right one based on config's type.
    # In Python, you'd use `isinstance(config, LeftCanonical)` branching inside one function.
    # Multiple dispatch is cleaner: each method is its own function definition.

    L = length(mps.tensors)
    tensors = copy(mps.tensors)     # SHALLOW copy: creates a new Vector, but QTensors inside
    bond_svs = copy(mps.bond_svs)   # are the same objects. The sweep helpers REPLACE tensors[i],
    # so they don't mutate the originals — they create new QTensor objects and assign them in.
    # In Python: tensors = mps.tensors[:] (shallow list copy).

    ε = _left_sweep_mps!(tensors, bond_svs, 1:(L - 1), config.trunc)
    # Sweep sites 1 through L-1 (not the last site — it becomes the norm carrier).
    # `1:(L-1)` = Python's `range(1, L)` — but 1-indexed and inclusive: [1, 2, …, L-1].
    # The `!` function mutates `tensors` and `bond_svs` in place; ε is the total truncation error.

    return FiniteMPS(tensors, bond_svs, CanonicalForm(L, L + 1), hypot(mps.ε, ε))
    # Tag: llim=L (sites 1..L-1 are left-canonical), rlim=L+1 (no right-canonical sites).
    # `hypot(mps.ε, ε)` ACCUMULATES onto whatever the input already carried. canonicalize is
    # a transformer, not a constructor: re-gauging cannot undo an approximation made earlier,
    # so the incoming error must survive. (Contrast `to_mps`, which builds a fresh MPS from a
    # dense state and correctly starts its accounting at zero.)
end

function canonicalize(mps::FiniteMPS, config::RightCanonical)
    # Mirror of LeftCanonical: full right sweep, norm accumulates at site 1.
    L = length(mps.tensors)
    tensors = copy(mps.tensors)
    bond_svs = copy(mps.bond_svs)
    ε = _right_sweep_mps!(tensors, bond_svs, 2:L, config.trunc)
    # Sweep sites 2 through L (not site 1 — it becomes the norm carrier).
    # `2:L` = Python's `range(2, L+1)` but reversed by `reverse()` inside the sweep helper.
    return FiniteMPS(tensors, bond_svs, CanonicalForm(0, 1), hypot(mps.ε, ε))   # accumulate onto the input's ε
    # Tag: llim=0 (no left-canonical sites — sentinel), rlim=1 (sites 1..L are right-canonical).
end

function canonicalize(mps::FiniteMPS, config::BondCanonical)
    L = length(mps.tensors)
    k = config.k   # the site that will hold the orthogonality centre
    tensors = copy(mps.tensors)
    bond_svs = copy(mps.bond_svs)

    # Left sweep: make sites 1..k-1 left-canonical.
    # `(k > 1) ? ... : 0.0` is ternary: if k=1 there are no left sites to sweep, so ε=0.
    ε = (k > 1) ? _left_sweep_mps!(tensors, bond_svs, 1:(k - 1), config.trunc) : 0.0

    # Right sweep: make sites k+1..L right-canonical.
    # If k=L there are no right sites to sweep.
    ε = hypot(
        ε, (k < L) ? _right_sweep_mps!(tensors, bond_svs, (k + 1):L, config.trunc) : 0.0
    )
    # The two sweeps touch disjoint bonds, so their discarded weights add: combine the
    # 2-norms in quadrature, the same rule used inside each sweep.

    _tag_centre!(tensors, k)
    # Relabel site k's index variances so that all legs are Upper (OC convention).
    # This is a pure metadata change — no data is modified.

    return FiniteMPS(tensors, bond_svs, CanonicalForm(k, k + 1), hypot(mps.ε, ε))   # accumulate onto the input's ε
    # Tag: llim=k (sites 1..k-1 left-canonical), rlim=k+1 (sites k+1..L right-canonical).
    # The OC is at site k, which is in the range [llim, rlim-1] = [k, k] — just site k.
end

function canonicalize(mps::FiniteMPS, config::SiteCanonical)
    L = length(mps.tensors)
    k = config.k
    tensors = copy(mps.tensors)
    bond_svs = copy(mps.bond_svs)

    # Same two-sweep structure as BondCanonical — the difference is in the form tag below.
    ε = (k > 1) ? _left_sweep_mps!(tensors, bond_svs, 1:(k - 1), config.trunc) : 0.0
    ε = hypot(
        ε, (k < L) ? _right_sweep_mps!(tensors, bond_svs, (k + 1):L, config.trunc) : 0.0
    )
    _tag_centre!(tensors, k)

    return FiniteMPS(tensors, bond_svs, CanonicalForm(k - 1, k + 1), hypot(mps.ε, ε))   # accumulate onto the input's ε
    # Tag: llim=k-1 (sites 1..k-2 left-canonical), rlim=k+1 (sites k+1..L right-canonical).
    # Site k is the orthogonality centre, EXCLUDED from both isometry ranges.
    # Difference from BondCanonical: CanonicalForm(k-1, k+1) vs CanonicalForm(k, k+1).
    # The extra site k is the un-factored centre tensor, not a diagonal SV matrix.
end

# SECTION -  canonical_error / is_canonical 

"""
    canonical_error(A::AbstractArray{<:Number,3}) -> Float64

Measure how far a rank-3 site tensor deviates from **left**-isometry:

```math
\\text{err} = \\|A^\\dagger A - I\\|_F
```

where ``A`` is first reshaped from ``(\\chi_L, d, \\chi_R)`` to the matrix
``(\\chi_L d \\times \\chi_R)``.

# Physical significance

The isometry condition ``A^\\dagger A = I`` is what makes expectation values
computable in ``O(L)`` time instead of ``O(d^L)``.  When the bra and ket MPS
are contracted site by site from the left, the ``A^\\dagger A`` pair at site
``i`` collapses to the identity, so only the open right index survives.
`canonical_error` is the MPS "health metric" — it quantifies how much that
collapse deviates from exact cancellation.

!!! note "Left isometry only"

    This function checks ``A^\\dagger A = I`` (left-canonical condition) only.
    Right-isometry ``BB^\\dagger = I`` is checked internally by [`is_canonical`](@ref)
    but not exposed through this API.  A right-facing overload can be added if needed.
"""
function canonical_error(A::AbstractArray{<:Number,3})
    # `AbstractArray{<:Number,3}` is a type constraint meaning:
    #   "a 3-dimensional array whose element type is a subtype of Number"
    # `<:Number` = "any numeric type" (Int, Float64, Complex128, etc.)
    # In Python this would be a type hint: `A: np.ndarray` (3D, numeric dtype).
    # Using AbstractArray lets this work on both Array (dense) and SubArray (array views).

    @debug "canonical_error: checking left isometry (A†A = I) only — right-isometry check not yet exposed"
    # `@debug` is a Julia MACRO (like Python's logging.debug). Macros start with `@` in Julia.
    # This message is only emitted when Julia's debug logging is enabled — zero cost otherwise.

    χL, d, χR = size(A)
    M = reshape(A, χL * d, χR)
    # Flatten the left and physical indices together to form an (χL*d × χR) matrix.
    # Physics: this is how we view A as a map from the right virtual space to (left⊗physical) space.

    return norm(M' * M - I(χR))
    # `M'` is Julia's adjoint (conjugate transpose): like numpy's `M.conj().T`.
    # `M' * M` = A†A reshaped, which should equal I(χR) for a left-canonical tensor.
    # `I(χR)` = χR×χR identity matrix — like numpy's `np.eye(χR)`.
    # `norm(...)` = Frobenius norm of the matrix (sum of squared elements, then sqrt).
    # Physics: ||A†A - I||_F = 0 iff A is exactly left-isometric. Any non-zero value
    # means the tensor has been perturbed or the canonical sweep was not exact.
end

# Private: right-isometry error used by is_canonical; not part of the public API.
function _right_isometry_error_mps(B::AbstractArray{<:Number,3})
    # Mirror of canonical_error but for RIGHT isometry: BB† = I.
    # Physics: B right-isometric means the rows of B (reshaped) are orthonormal.
    # This is checked for sites to the RIGHT of the orthogonality centre.
    χL, d, χR = size(B)
    M = reshape(B, χL, d * χR)
    # For right isometry: group the right indices (physical + right virtual) together.
    # Rows of M should be orthonormal: MM† = I(χL).
    return norm(M * M' - I(χL))
    # `M * M'` = BB† (as a matrix product). Compare: canonical_error uses M'*M (left).
    # Returns Frobenius norm of deviation from the identity — zero for a perfect right-isometry.
end

"""
    is_canonical(ψ::FiniteMPS; tol = 1e-10) -> Bool

Return `true` if every site tensor in `ψ` satisfies its expected isometry
condition as given by `ψ.form`:

| Form tag                    | Sites checked                                     |
|:--------------------------- |:------------------------------------------------- |
| `CanonicalForm(llim, rlim)` | 1..llim-1 left-isometric; rlim..L right-isometric |
| `VidalForm()`               | always `true`                                     |
| `ArbitraryForm()`           | always `false`                                    |

The centre site(s) between `llim` and `rlim` (if any) are **not** checked —
they hold the gauge weight and need not be isometric.
"""
function is_canonical(ψ::FiniteMPS; tol::Float64=1e-10)
    # `tol::Float64=1e-10` is a keyword argument with a DEFAULT value of 1e-10.
    # Called as `is_canonical(ψ)` or `is_canonical(ψ; tol=1e-8)`.
    # The `;` in Julia function signatures separates positional from keyword arguments.
    # In Python: `def is_canonical(ψ, *, tol=1e-10):`.

    L = length(ψ.tensors)
    form = ψ.form   # extract the form tag (CanonicalForm, VidalForm, or ArbitraryForm)

    if form isa CanonicalForm
        # `isa` in Julia = `isinstance` in Python. Checks if `form` is a CanonicalForm instance.
        llim, rlim = form.llim, form.rlim
        # Destructure the struct fields. Same as Python: `llim, rlim = form.llim, form.rlim`.

        # Check left isometry for sites 1..llim-1
        for i in 1:(llim - 1)
            canonical_error(ψ.tensors[i].data) > tol && return false
            # `expr && return false` = short-circuit: if canonical_error > tol, immediately return false.
            # Equivalent to Python's `if canonical_error(...) > tol: return False`.
            # The `&&` is Julia's short-circuit AND — the right side only runs if left side is true.
        end

        # Check right isometry for sites rlim..L
        for i in rlim:L
            _right_isometry_error_mps(ψ.tensors[i].data) > tol && return false
            # Same pattern: if any right-isometry check fails, return false immediately.
        end

        return true   # all checks passed — the MPS is in the canonical form it claims to be
    elseif form isa VidalForm
        return true   # Vidal form is always considered "canonical" (it has its own invariants)
    else
        return false  # ArbitraryForm: no isometry guaranteed → not canonical
    end
end
