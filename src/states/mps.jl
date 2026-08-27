"""
    AbstractMPSForm

Supertype for all MPS gauge-form tags. Every `FiniteMPS` carries one of these
to record which isometry conditions its site tensors currently satisfy.
Concrete subtypes: [`CanonicalForm`](@ref), [`VidalForm`](@ref), [`ArbitraryForm`](@ref).
"""
# `abstract type` = Python ABC. This is the shared parent for all "form tag" types.
# Form tags are metadata: they tell you WHAT isometry guarantees the MPS currently holds.
# Julia uses these for dispatch — functions can branch on the form tag's type.
abstract type AbstractMPSForm end

"""
    CanonicalForm(llim, rlim)

Canonical form tag for a finite MPS.

- `llim::Int`: first site of the orthogonality-centre region; sites ``1, \\ldots, \\texttt{llim}-1``
  are left-isometric (``A_i^\\dagger A_i = I``).
- `rlim::Int`: first right-isometric site; sites ``\\texttt{rlim}, \\ldots, L`` satisfy
  ``B_i B_i^\\dagger = I``.

The OC region occupies sites ``\\texttt{llim} \\ldots \\texttt{rlim}-1``; those tensors hold the
gauge weight and are **not** checked by [`is_canonical`](@ref).  The values ``\\texttt{llim}=0``
and ``\\texttt{rlim}=L+1`` are out-of-bounds sentinels that make the corresponding range empty
without any special-case branching.

| Form | `llim` | `rlim` | OC region |
|------|--------|--------|-----------|
| Fully left-canonical ([`LeftCanonical`](@ref))  | ``L``   | ``L+1`` | site ``L`` (sentinel: no right-canonical sites) |
| Fully right-canonical ([`RightCanonical`](@ref)) | ``0``   | ``1``   | — (sentinel: no left-canonical sites) |
| Bond canonical at bond ``k`` ([`BondCanonical`](@ref)) | ``k`` | ``k+1`` | site ``k`` holds the singular values |
| Site canonical at site ``k`` ([`SiteCanonical`](@ref)) | ``k-1`` | ``k+1`` | site ``k`` is the un-gauged centre tensor |

!!! note "Mixed canonical vs bond canonical"
    What the MPS literature calls *mixed canonical form* is bond canonical — singular values
    sit on a specific bond.  Site canonical is a related but distinct gauging where the centre
    tensor itself carries the full weight.
"""
# This struct encodes the canonical form as TWO integers (llim, rlim).
# Why two integers? Because the OC can span multiple sites (though usually it's just one).
# `Int` in Julia is like Python's `int`. The `::Int` syntax is a type annotation on the field.
# Comparing forms is done with `==` which Julia auto-defines for plain structs (value equality,
# just like Python dataclasses with `eq=True`).
struct CanonicalForm <: AbstractMPSForm
    llim::Int  # left limit: sites 1..(llim-1) are left-canonical (A†A = I)
    rlim::Int  # right limit: sites rlim..L are right-canonical (BB† = I)
end

"""
    VidalForm()

Form tag indicating the MPS is in Vidal's ``\\Gamma\\Lambda`` representation.
Site tensors are stored as ``\\Gamma_i`` and bond tensors as ``\\Lambda_i``.
[`is_canonical`](@ref) returns `true` for this form without checking isometry.
"""
# A struct with NO fields acts like Python's `class VidalForm: pass` — a singleton tag.
# In Julia, `VidalForm()` constructs the single instance. You can compare: `x isa VidalForm`.
# Physics: Vidal form separates the state into Gamma tensors (gauge-fixed) and Lambda vectors
# (Schmidt values on each bond). This is the standard TEBD representation.
struct VidalForm <: AbstractMPSForm end

"""
    ArbitraryForm()

Form tag indicating no isometry conditions are guaranteed — the MPS has been
modified (e.g. by [`add_mps`](@ref)) without a subsequent canonicalization.
[`is_canonical`](@ref) returns `false` for this form.
"""
# Another empty struct. Used as a "dirty flag" after operations like add_mps that
# break canonical form but don't immediately re-canonicalize.
struct ArbitraryForm <: AbstractMPSForm end

"""
    FiniteMPS(tensors, bond_svs, form, ε)

Matrix-product state for a finite open chain with ``L`` sites.

# Fields
- `tensors::Vector{QTensor}`: ``L`` site tensors, each valence-3 with legs stored
  in the order ``(\\texttt{vL},\\; \\sigma,\\; \\texttt{vR})``
- `bond_svs::Vector{SingValSpectrum}`: ``L+1`` bond spectra; boundaries carry the
  trivial spectrum ``[1.0]``
- `form::AbstractMPSForm`: canonical-form tag
- `ε::Float64`: accumulated truncation error — the per-bond discarded singular-value 2-norms
  combined **in quadrature**, ``\\varepsilon = \\sqrt{\\sum_k \\varepsilon_k^2}`` (zero for
  `NoTrunc`)

# What `ε` means

``\\varepsilon^2`` is the total **discarded weight**: the Schmidt weight thrown away by every
truncation the state has been through. For real-time evolution this has a directly measurable
counterpart — the evolution is unitary, so the only way the state can lose norm is truncation,
and therefore

```math
\\varepsilon^2 = 1 - \\lVert\\psi\\rVert^2
```

for a state that started unit-norm and was never renormalized. That identity is the practical
way to sanity-check a run.

!!! warning "Not a rigorous upper bound"
    Quadrature accumulation is exact when successive truncations discard *independent*
    directions, and can **under**-estimate when they are correlated. Plain summation of the
    per-bond ``\\varepsilon_k`` would be a rigorous bound by the triangle inequality, but over a
    long run it drifts to O(1) regardless of the true error and stops being a usable
    diagnostic. Quadrature trades the guarantee for a number that tracks reality.

# Index convention (MasterPlan §13/§23; von Delft covariant notation)

The physical leg ``\\sigma`` is **always `Upper`** — the stored array is the
contravariant ket-expansion coefficient ``A^{\\sigma}``. Bond variance is
**form-dependent**: every bond arrow points toward the orthogonality centre
(`Upper` = arrow in, `Lower` = arrow out), so

| Site | vL | ``\\sigma`` | vR |
|------|----|----|----|
| left-canonical ``A^{i,\\sigma}_k`` | `Upper` | `Upper` | `Lower` |
| right-canonical ``B_k^{i,\\sigma}`` | `Lower` | `Upper` | `Upper` |
| orthogonality centre | `Upper` | `Upper` | `Upper` |

Boundary sites have ``\\chi_L = 1`` (left) and ``\\chi_R = 1`` (right).
"""
# The main MPS container struct. In Python you'd write this as a dataclass:
#   @dataclass(frozen=True)  # frozen because Julia structs are immutable by default
#   class FiniteMPS:
#       tensors: List[QTensor]       # L site tensors
#       bond_svs: List[SingValSpectrum]  # L+1 bond spectra (one more than sites!)
#       form: AbstractMPSForm        # which isometry guarantee holds right now
#       ε: float                     # total truncation error accumulated so far
#
# `Vector{QTensor}` = Python's `List[QTensor]`. Julia uses `Vector` for 1D arrays.
# `Float64` = Python's `float` (IEEE 754 double precision, 64-bit).
struct FiniteMPS
    tensors::Vector{QTensor}           # L valence-3 site tensors: each is (χL, σ, χR)
    bond_svs::Vector{SingValSpectrum}  # L+1 bond singular-value spectra (boundaries included)
    form::AbstractMPSForm              # which canonical form currently holds
    ε::Float64                         # accumulated truncation error (0.0 if no truncation)
end

# SECTION -  Internal helpers 
# These two functions (_left_sweep, _right_sweep) do the heavy lifting of
# converting a DENSE state tensor ψ (shape d^L) into an MPS via iterated SVD.
# They are "internal" (prefixed with _) — callers use to_mps() below.

function _left_sweep(ψ::QTensor, d::Vector{Int}, trunc::AbstractTrunc)
    # ψ::QTensor means the argument must be a QTensor (Julia type annotation on function args).
    # d::Vector{Int} is a list of local Hilbert-space dimensions, e.g. [2,2,2,2] for 4 qubits.
    # trunc::AbstractTrunc is the truncation strategy (NoTrunc(), MaxBondDimTrunc(D), etc.)
    # Physics: a left sweep means we sweep left→right, making each site left-canonical (A†A=I),
    # and the norm accumulates at the rightmost site.

    L = length(d)  # number of sites in the chain; `length` in Julia = Python's `len`

    # Pre-allocate output arrays. `Vector{QTensor}(undef, L)` creates a length-L array of
    # QTensor objects WITHOUT initializing the entries — like numpy's np.empty(L, dtype=object).
    # `undef` is Julia's sentinel for "uninitialized" — accessing before writing would be an error.
    tensors = Vector{QTensor}(undef, L)
    bond_svs = Vector{SingValSpectrum}(undef, L + 1)  # L+1 bonds for L sites (open chain)
    ε_total = 0.0  # running sum of truncation errors across all bonds

    # Set up trivial boundary spectra. For an open chain, the left boundary is
    # a virtual index of dimension 1 with a single singular value of 1.0.
    # `SingValSpectrum([1.0], 0.0, true)` = spectrum with values=[1.0], error=0.0, normalized=true.
    bond_svs[1] = SingValSpectrum([1.0], 0.0, true)   # left boundary: trivial χ=1
    bond_svs[L + 1] = SingValSpectrum([1.0], 0.0, true)   # right boundary: trivial χ=1

    # The "carry" is the part of the state tensor we haven't yet decomposed.
    # Initially it is the FULL state tensor reshaped to (1, d₁, d₂, …, d_L).
    # `reshape(ψ.data, 1, d...)` uses the `d...` "splat" operator — like Python's `*d`.
    # The leading 1 is the dummy left virtual index (boundary condition: χ_left=1).
    # Physics: we think of the full state as a 1×d^L matrix with a trivial left index.
    carry = reshape(ψ.data, 1, d...)  # shape: (1, d₁, d₂, …, d_L)
    χ_left = 1                         # current left bond dimension (starts trivial)

    # Loop from site 1 to L-1 (the last site is handled separately after the loop).
    # Julia's `1:(L-1)` is a range object like Python's `range(1, L)` — but 1-indexed!
    for i in 1:(L - 1)
        # Physics: at each bond cut i|i+1, we reshape the carry into a matrix:
        #   left group  = (χ_left × d_i)   — all indices to the LEFT of the cut
        #   right group = (d_{i+1} × … × d_L)  — all indices to the RIGHT
        # This is the matrix form needed for SVD.
        right_dim = prod(d[(i + 1):end])  # product of all remaining local dimensions
        # `d[(i+1):end]` = Python's `d[i:]` (slice to end). `end` is Julia's equivalent of `-1`/`len`.
        # `prod(...)` = Python's `math.prod(...)` or `np.prod(...)`.

        M = reshape(carry, χ_left * d[i], right_dim)
        # reshape to matrix: rows = (χ_left × d_i), columns = (product of right dims)
        # In numpy: M = carry.reshape(χ_left * d[i], right_dim)

        # Compute the full SVD: M = U × Diagonal(S) × Vt
        # `F` is a struct with fields F.U, F.S, F.Vt (note: Vt = V†, the conjugate transpose).
        # In numpy: U, S, Vt = np.linalg.svd(M, full_matrices=False)
        F = _robust_svd(M)

        # Noise-cleaning threshold (Golub–Van Loan criterion):
        # Tiny floating-point artifacts in F.S at level ε_machine × σ_max × n
        # would inflate χ if not removed before truncation. This threshold
        # is the standard numerical-rank cutoff from numerical linear algebra.
        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        # `eps(eltype(F.S))` = machine epsilon for the float type (≈1.1e-16 for Float64)
        # `eltype(F.S)` = Python's `F.S.dtype` — the element type of the array
        # `isempty(F.S)` = Python's `len(F.S) == 0` — guard against empty singular value list
        # `F.S[1]` = the largest singular value (Julia indexing: 1-based, not 0-based!)

        S_clean = filter(s -> s > tol, F.S)
        # `filter(predicate, collection)` = Python's `[s for s in F.S if s > tol]`
        # `s -> s > tol` is an anonymous function (lambda): like Python's `lambda s: s > tol`

        r, ε_bond = _truncate_singular_values(S_clean, trunc)
        # `_truncate_singular_values` returns (r, ε_bond):
        #   r = number of singular values to KEEP after applying truncation strategy
        #   ε_bond = the 2-norm of discarded singular values (truncation error at this bond)
        # Julia allows multiple return values as a tuple — like Python's `r, eps = func(...)`.

        svs = F.S[1:r]   # keep only the top r singular values; F.S[1:r] = Python's F.S[:r]
        ε_total = hypot(ε_total, ε_bond)
        # Accumulate in QUADRATURE: ε is a 2-norm, so ε² is the weight discarded at this
        # bond. The WEIGHTS are what add across bonds, so the norms combine as sqrt(a²+b²).
        # `hypot` computes that without overflowing on the intermediate squares.

        # Build the left-canonical site tensor A_i from U's first r columns:
        # U has shape (χ_left×d[i], r); reshape to (χ_left, d[i], r) for site tensor.
        # Physics: U has orthonormal columns → A_i†A_i = I (left-isometry condition).
        tensors[i] = QTensor(
            reshape(F.U[:, 1:r], χ_left, d[i], r),
            (upper(:vL, χ_left), upper(:σ, d[i]), lower(:vR, r)),
        )
        # `upper(:vL, χ_left)` creates an Upper-variance tensor index named :vL with dimension χ_left.
        # `:vL` is a Julia Symbol (like Python's string "vL" but used as an identifier).
        # `lower(:vR, r)` is a Lower-variance index (arrow pointing OUT from left-canonical tensor).
        # Physics: for a left-canonical tensor, the right bond arrow points away from the OC.

        # Check if the kept singular values are normalized (||svs||² ≈ 1).
        # This is TRUE when the input MPS was normalized AND no significant truncation occurred.
        normalized = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        # `isapprox(a, b; atol=...)` = Python's `np.isclose(a, b, atol=...)` but returns Bool
        # `sum(abs2, svs)` = sum of squares of singular values = Python's `np.sum(svs**2)`
        # `abs2(x)` = |x|² (slightly faster than `abs(x)^2` since it avoids a sqrt)

        bond_svs[i + 1] = SingValSpectrum(svs, ε_bond, normalized)
        # Store the Schmidt spectrum at bond i|i+1.

        # Form the new carry: Σ·V† absorbed into the next site's data.
        # Physics: the carry is the RIGHT factor (Σ·V†), which flows rightward.
        # After the last site this carry becomes the norm carrier (un-normalized final tensor).
        carry = reshape(Diagonal(svs) * F.Vt[1:r, :], r, d[(i + 1):end]...)
        # `Diagonal(svs)` creates an r×r diagonal matrix from the vector svs — like np.diag(svs).
        # `F.Vt[1:r, :]` = first r rows of V† (shape: r × right_dim)
        # `Diagonal(svs) * F.Vt[1:r, :]` is a matrix product: (r×r)×(r×right_dim) → (r×right_dim)
        # `reshape(..., r, d[(i+1):end]...)` unpacks right_dim back into (d_{i+1}, …, d_L) axes.

        χ_left = r  # the new left bond dimension for the next iteration is r (truncated rank)
    end

    # Last site: the loop ended at i=L-1, so carry now has shape (χ_left, d_L).
    # The last site tensor is just this carry reshaped to (χ_left, d_L, 1) — the trailing 1
    # is the trivial right boundary bond (χ_R=1 on an open chain).
    tensors[L] = QTensor(
        reshape(carry, χ_left, d[L], 1),
        (upper(:vL, χ_left), upper(:σ, d[L]), lower(:vR, 1)),
    )
    # Physics: the last site is NOT constrained to be left-isometric — it holds the full norm.
    # Both vL and σ are Upper; vR is Lower (direction convention for left-canonical tensors).

    # Construct and return the FiniteMPS. `CanonicalForm(L, L+1)` means:
    #   llim=L → sites 1..L-1 are left-canonical
    #   rlim=L+1 → no sites are right-canonical (L+1 is out-of-bounds sentinel)
    return FiniteMPS(tensors, bond_svs, CanonicalForm(L, L + 1), ε_total)
end

function _right_sweep(ψ::QTensor, d::Vector{Int}, trunc::AbstractTrunc)
    # Mirror of _left_sweep: sweeps right→left, making each site right-canonical (BB†=I).
    # Physics: the norm accumulates at the LEFTMOST site (site 1) instead of the rightmost.
    L = length(d)
    tensors = Vector{QTensor}(undef, L)
    bond_svs = Vector{SingValSpectrum}(undef, L + 1)
    ε_total = 0.0

    bond_svs[1] = SingValSpectrum([1.0], 0.0, true)
    bond_svs[L + 1] = SingValSpectrum([1.0], 0.0, true)

    # Initial carry has the full state tensor with a trailing trivial dimension 1 (right boundary).
    # Compare to _left_sweep which prepended the 1; here we APPEND it.
    carry = reshape(ψ.data, d..., 1)  # shape: (d₁, …, d_L, 1)
    χ_right = 1                          # current right bond dimension (starts trivial)

    # Loop from site L down to site 2 (site 1 is handled after the loop).
    # `L:-1:2` is Julia's reverse range — like Python's `range(L, 1, -1)`.
    for i in L:-1:2
        # Physics: at bond cut (i-1)|i, we reshape carry so:
        #   left group  = (d_1 × … × d_{i-1})   — all LEFT of cut
        #   right group = (d_i × χ_right)         — current site + already-processed right part
        left_dim = prod(d[1:(i - 1)])      # product of all already-processed left dimensions
        M = reshape(carry, left_dim, d[i] * χ_right)
        # In numpy: M = carry.reshape(left_dim, d[i] * χ_right)
        # Matrix: rows = left_dim, columns = (d_i × χ_right)

        F = _robust_svd(M)
        # Same noise-cleaning threshold as the left sweep.
        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        S_clean = filter(s -> s > tol, F.S)
        r, ε_bond = _truncate_singular_values(S_clean, trunc)

        svs = F.S[1:r]
        ε_total = hypot(ε_total, ε_bond)   # quadrature: discarded WEIGHTS add, so the 2-norms combine as sqrt(a²+b²)

        # Build the right-canonical site tensor B_i from V†'s first r rows:
        # Vt has shape (r, d[i]×χ_right); reshape to (r, d[i], χ_right).
        # Physics: V† has orthonormal rows → B_i·B_i† = I (right-isometry condition).
        tensors[i] = QTensor(
            reshape(F.Vt[1:r, :], r, d[i], χ_right),
            (lower(:vL, r), upper(:σ, d[i]), upper(:vR, χ_right)),
        )
        # For right-canonical tensors: vL is Lower (arrow points LEFT, away from OC),
        # σ is always Upper (contravariant physical leg), vR is Upper (toward OC on the right).

        normalized = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        bond_svs[i] = SingValSpectrum(svs, ε_bond, normalized)
        # Note: for the right sweep, bond_svs[i] corresponds to the bond to the LEFT of site i
        # (between sites i-1 and i). This is indexed differently from the left sweep's bond_svs[i+1].

        # Form the new carry: U·Σ, which flows LEFTWARD to be absorbed into site i-1.
        # Physics: U·Σ is the left factor, carrying the gauge weight leftward.
        carry = reshape(F.U[:, 1:r] * Diagonal(svs), d[1:(i - 1)]..., r)
        # `F.U[:, 1:r]` = first r columns of U (shape: left_dim × r)
        # `F.U[:, 1:r] * Diagonal(svs)` = (left_dim×r) × (r×r) → (left_dim×r)
        # Then reshape to (d_1, …, d_{i-1}, r) — expanding the left_dim product back into individual dims.

        χ_right = r  # the new right bond dimension for the next iteration
    end

    # Site 1: carry has shape (d₁, χ_right). Prepend a trivial left dim 1 (boundary condition).
    tensors[1] = QTensor(
        reshape(carry, 1, d[1], χ_right),
        (lower(:vL, 1), upper(:σ, d[1]), upper(:vR, χ_right)),
    )
    # Physics: site 1 is the norm carrier in right-canonical form. vL is Lower (pointing left),
    # which is the convention for right-canonical tensors — arrow points away from the OC.

    # `CanonicalForm(0, 1)`: llim=0 (sentinel: no left-canonical sites), rlim=1 (all sites right-canonical).
    return FiniteMPS(tensors, bond_svs, CanonicalForm(0, 1), ε_total)
end

# SECTION -  Public API 

"""
    to_mps(ψ::QTensor; trunc::AbstractTrunc = NoTrunc(), form::Symbol = :left) -> FiniteMPS

Decompose a full quantum state tensor into a canonical matrix-product state via
iterated SVD.

# Arguments

  - `ψ::QTensor`: full coefficient tensor ``A^{\\sigma_1 \\ldots \\sigma_L}`` with
    ``L`` physical legs, all of type `Upper` (contravariant ket-expansion indices)
  - `trunc::AbstractTrunc`: truncation strategy (default: `NoTrunc()`)
  - `form::Symbol`: `:left` for left-canonical sweep or `:right` for right-canonical sweep

# Physical invariants

| Property     | Left                                    | Right                                 |
|:------------ |:--------------------------------------- |:------------------------------------- |
| Isometry     | ``A_i^\\dagger A_i = I`` (sites 1..L-1) | ``B_i B_i^\\dagger = I`` (sites 2..L) |
| Form tag     | `CanonicalForm(L, L+1)`                 | `CanonicalForm(0, 1)`                 |
| Norm carrier | last site                               | first site                            |

  - **Reconstruction**: contracting all tensors recovers ``\\psi`` within `mps.ε`
  - **Error accounting**: `mps.ε = \\sqrt{\\sum_i \\texttt{bond\\_svs}[i].\\varepsilon^2}` — the
    per-bond errors are 2-norms of discarded singular values, so it is their *squares* (the
    discarded weights) that add. Note this is not a rigorous upper bound; see [`FiniteMPS`](@ref).
  - **Boundary spectra**: `bond_svs[1] = bond_svs[L+1] = [1.0]`

# Algorithm: carry-propagation via iterated SVD

**Left sweep.** At each bond cut ``i``, the carry tensor (shape
``(\\chi_{i-1},\\, d_i,\\, d_{i+1},\\ldots,d_L)``) is reshaped into a matrix

```math
M = \\operatorname{reshape}(\\texttt{carry},\\; \\chi_{i-1} d_i,\\; d_{i+1}\\cdots d_L)
```

and factored as ``M = U\\Sigma V^\\dagger``.  ``U`` (shape ``\\chi_{i-1} d_i \\times r``,
isometric columns) becomes site tensor ``A_i``, while ``\\Sigma V^\\dagger`` becomes
the new carry.  Because ``U`` has orthonormal columns, ``A_i^\\dagger A_i = I_r``
automatically.  The norm of ``\\psi`` rides rightward in the carry and ends up
absorbed into the last site.

**Right sweep.** The right sweep is the exact mirror: the carry propagates
leftward as ``U\\Sigma``, and ``V^\\dagger`` (rows orthonormal) becomes site
tensor ``B_i``, satisfying ``B_i B_i^\\dagger = I_r``.  Site 1 inherits the norm.

**Noise cleaning before truncation.** `LinearAlgebra.svd` can return tiny
floating-point artifacts in the singular-value tail at the level of
``\\varepsilon_\\text{mach} \\times \\sigma_1``.  These are filtered out before
`_truncate_singular_values` is called, preventing them from being counted
as genuine Schmidt values and inflating ``\\chi`` when `trunc = NoTrunc()`.
The threshold used is the classical Golub–Van Loan numerical-rank criterion:
``\\sigma_k > n\\, \\varepsilon_\\text{mach}\\, \\sigma_1``.
"""
function to_mps(ψ::QTensor; trunc::AbstractTrunc=NoTrunc(), form::Symbol=:left)::FiniteMPS
    # `::FiniteMPS` after the `)` is the RETURN TYPE annotation. Julia checks this at runtime.
    # In Python: `def to_mps(ψ, *, trunc=NoTrunc(), form=':left') -> FiniteMPS:`
    # The `;` in the argument list marks keyword-only arguments (must be passed as `form=:left`).

    L = ndims(ψ.data)   # number of tensor dimensions = number of sites L
    # `ndims` = Python's `ψ.data.ndim` (numpy array attribute). Each leg = one site.

    d = [dim(ψ.indices[i]) for i in 1:L]
    # List comprehension — same as Python's `[dim(ψ.indices[i]) for i in range(1, L+1)]`.
    # `dim(index)` extracts the dimension of a tensor index (local Hilbert-space size d_i).
    # Result: a Vector{Int} of local dimensions, e.g. [2, 2, 2, 2] for 4 qubits.

    if form === :left
        # `===` in Julia is identity comparison for immutables (Symbols are immutable,
        # so this is equivalent to `==` here). `:left` is a Julia Symbol — like "left" in Python
        # but interned and compared by identity, slightly faster than string comparison.
        return _left_sweep(ψ, d, trunc)
    elseif form === :right
        return _right_sweep(ψ, d, trunc)
    else
        # `throw(ArgumentError(...))` = Python's `raise ValueError(...)`.
        # `$form` inside a string is Julia's string interpolation — like Python's f"{form}".
        throw(ArgumentError("to_mps: form must be :left or :right, got $form"))
    end
end

# SECTION -  MPS addition 

"""
    add_mps(a, ψ::FiniteMPS, b, φ::FiniteMPS; trunc=NoTrunc()) -> FiniteMPS

Compute the superposition ``a|\\psi\\rangle + b|\\varphi\\rangle`` as a new MPS.

The result is built by the direct-sum (block-diagonal) construction: at each
interior site the virtual bond is split into a block for ``|\\psi\\rangle`` and a
block for ``|\\varphi\\rangle``, giving bond dimension ``\\chi_\\psi + \\chi_\\varphi``.
The boundary conditions stitch the two chains together so the boundary tensors
absorb the coefficients ``a`` and ``b``.

After the block-diagonal assembly, a left-to-right re-canonicalization sweep
with `trunc` is applied to compress the bond dimension and produce a
`CanonicalForm(L, L+1)` result.

# Arguments

  - `a`, `b` — scalar coefficients (zero coefficient → zero contribution)
  - `ψ`, `φ`  — input MPS (must have the same length and physical dimensions)
  - `trunc`   — truncation strategy applied during the recompression sweep

# Returns

A left-canonical `FiniteMPS` representing ``a|\\psi\\rangle + b|\\varphi\\rangle``.

# See also

`Base.:+(ψ, φ)` — unit-coefficient shorthand.
"""
function add_mps(
    a::Number, ψ::FiniteMPS, b::Number, φ::FiniteMPS; trunc::AbstractTrunc=NoTrunc()
)::FiniteMPS
    # `Number` is Julia's abstract type for all numeric types (Int, Float64, Complex, etc.)
    # This is like Python's `Union[int, float, complex]`. Julia dispatches automatically.

    L = length(ψ.tensors)   # number of sites in ψ (and φ must match)
    L == length(φ.tensors) || throw(
        ArgumentError("add_mps: MPS lengths must match, got $L and $(length(φ.tensors))"),
    )
    # `a || throw(...)` is Julia's short-circuit "or" guard — if `a` is false, throw.
    # Equivalent to Python's `assert a, "..."` but raises ArgumentError not AssertionError.
    # `$(length(φ.tensors))` interpolates a function call result into the string.

    all(size(ψ.tensors[i].data, 2) == size(φ.tensors[i].data, 2) for i in 1:L) ||
        throw(ArgumentError("add_mps: physical dimensions must match at every site"))
    # `all(condition for i in range)` = Python's `all(cond(i) for i in range(1, L+1))`.
    # `size(A, 2)` = Python's `A.shape[1]` (2nd dimension, 0-indexed in Python but 1-indexed here).
    # Dimension 2 of the site tensor is the physical leg σ (convention: vL=1, σ=2, vR=3).

    # `promote_type(T1, T2, ...)` finds the common numeric type that can hold values of all given types.
    # Like Python's implicit numeric promotion: int + float → float. Here we make everything agree.
    T = promote_type(
        typeof(a), typeof(b), eltype(ψ.tensors[1].data), eltype(φ.tensors[1].data)
    )
    # `typeof(x)` = type(x) in Python. `eltype(A)` = A.dtype for arrays.

    # Build direct-sum site tensors (block-diagonal construction)
    tensors = Vector{QTensor}(undef, L)
    for i in 1:L
        Aψ = convert(Array{T}, ψ.tensors[i].data)  # (χLψ, d, χRψ)
        Aφ = convert(Array{T}, φ.tensors[i].data)  # (χLφ, d, χRφ)
        # `convert(Array{T}, x)` = like Python's `np.array(x, dtype=T)` — ensures same numeric type.
        # This is needed because a, b might be Float64 while tensor entries are Float32, etc.

        χLψ, d, χRψ = size(Aψ)   # destructure the 3-tuple of dimensions (like Python tuple unpacking)
        χLφ, _d, χRφ = size(Aφ)  # `_d` is convention for "unused variable" — we already know d

        # Absorb coefficients at boundary sites (only site 1):
        # Physics: the coefficient a or b must appear exactly once in the MPS product.
        # We choose to put it in the leftmost tensor.
        if i == 1
            Aψ = a .* Aψ   # `.* ` is elementwise multiplication (broadcast); like numpy's `a * Aψ`
            Aφ = b .* Aφ   # scales each element of the array by the scalar b
        end

        # The new bond dimensions after direct-summing:
        # At boundary sites (i=1 or i=L), the boundary virtual dim stays 1.
        # At interior sites, the direct sum DOUBLES the bond dim (χψ + χφ).
        χL_new = i == 1 ? 1 : χLψ + χLφ
        # `cond ? a : b` = Python's `a if cond else b` (ternary operator)
        χR_new = i == L ? 1 : χRψ + χRφ

        if i == 1
            # Left boundary site: both ψ and φ have χL=1, so the left dim is 1.
            # Direct sum happens in the RIGHT (vR) direction: stack Aψ and Aφ side-by-side.
            # Physics: |ψ⟩ and |φ⟩ live in different "channels" of the virtual index.
            blk = zeros(T, 1, d, χR_new)
            # `zeros(T, 1, d, χR_new)` = np.zeros((1, d, χR_new), dtype=T) — all zeros array
            blk[1, :, 1:χRψ] = Aψ[1, :, :]   # fill ψ-block (columns 1..χRψ)
            blk[1, :, (χRψ + 1):end] = Aφ[1, :, :]   # fill φ-block (columns χRψ+1..end)
        # `1:χRψ` = Python's `slice(0, χRψ)` = `[:χRψ]`. Julia is 1-indexed, inclusive on both ends.

        elseif i == L
            # Right boundary site: both have χR=1, so the right dim is 1.
            # Direct sum happens in the LEFT (vL) direction: stack Aψ and Aφ top-to-bottom.
            blk = zeros(T, χL_new, d, 1)
            blk[1:χLψ, :, 1] = Aψ[:, :, 1]   # fill ψ-block (rows 1..χLψ)
            blk[(χLψ + 1):end, :, 1] = Aφ[:, :, 1]   # fill φ-block (rows χLψ+1..end)

        else
            # Interior site: the block-diagonal structure is in (vL, vR).
            # Physics: the ψ-channel occupies the top-left block, φ-channel the bottom-right.
            # The off-diagonal blocks are zero (the two states don't mix).
            blk = zeros(T, χL_new, d, χR_new)
            blk[1:χLψ, :, 1:χRψ] = Aψ         # top-left block
            blk[(χLψ + 1):end, :, (χRψ + 1):end] = Aφ         # bottom-right block
            # The off-diagonal blocks remain zero — this is the "direct sum" structure.
        end

        # transient ArbitraryForm tensors: σ is Upper on every state tensor; the
        # bond tags are provisional until _recompress_left assigns the canonical ones
        new_indices = (upper(:vL, χL_new), upper(:σ, d), lower(:vR, χR_new))
        tensors[i] = QTensor(blk, new_indices)
    end

    # Build trivial boundary spectra — the block-diagonal MPS is in ArbitraryForm
    # `fill(val, n)` creates a length-n array filled with `val` — like Python's `[val] * n`
    bond_svs = fill(SingValSpectrum([1.0], 0.0, true), L + 1)
    # Tag as ArbitraryForm because the block-diagonal MPS is NOT canonical yet.
    raw = FiniteMPS(tensors, bond_svs, ArbitraryForm(), 0.0)

    # Recompress via left sweep to get a canonical form and apply truncation
    # This makes the bond dimension optimal (removes redundancy from the direct-sum structure).
    return _recompress_left(raw, trunc)
end

"""
    _recompress_left(mps, trunc) -> FiniteMPS

Internal helper: apply a left-to-right QR/SVD sweep to bring `mps` into
left-canonical form and apply `trunc` at each bond. Used by `add_mps` to
compress the block-diagonal superposition.

This is identical in spirit to `_left_sweep` but operates on existing
site tensors (no full state tensor to start from) by treating the first
tensor as the initial carry.
"""
function _recompress_left(mps::FiniteMPS, trunc::AbstractTrunc)::FiniteMPS
    L = length(mps.tensors)
    tensors = Vector{QTensor}(undef, L)
    bond_svs = Vector{SingValSpectrum}(undef, L + 1)
    ε_total = 0.0

    bond_svs[1] = SingValSpectrum([1.0], 0.0, true)
    bond_svs[L + 1] = SingValSpectrum([1.0], 0.0, true)

    # carry starts as the first site tensor, reshape to matrix (d, χR)
    carry = mps.tensors[1].data  # (1, d, χR)  — same shape as any left site tensor
    χL = size(carry, 1)          # `size(A, dim)` = Python's `A.shape[dim-1]` (1-indexed!)

    for i in 1:(L - 1)
        d = size(carry, 2)     # physical dimension of current site
        χR = size(carry, 3)     # right bond dimension of current site
        M = reshape(carry, χL * d, χR)   # flatten to matrix for SVD

        F = _robust_svd(M)
        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        S_clean = filter(s -> s > tol, F.S)
        r, ε_bond = _truncate_singular_values(S_clean, trunc)

        svs = F.S[1:r]
        ε_total = hypot(ε_total, ε_bond)   # quadrature: discarded WEIGHTS add, so the 2-norms combine as sqrt(a²+b²)

        tensors[i] = QTensor(
            reshape(F.U[:, 1:r], χL, d, r), (upper(:vL, χL), upper(:σ, d), lower(:vR, r))
        )
        normalized = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        bond_svs[i + 1] = SingValSpectrum(svs, ε_bond, normalized)

        # Next carry: Σ·Vt contracted with the next site tensor
        # This is the key difference from _left_sweep: instead of reshaping a global tensor,
        # we CONTRACT the carry with the next MPS site tensor.
        SV = Diagonal(svs) * F.Vt[1:r, :]   # (r, χR) — the "transfer" factor
        next = mps.tensors[i + 1].data       # (χR_old, d_next, χR_next) — next site's raw data
        d_next = size(next, 2)
        χR_next = size(next, 3)
        # Contract SV (r×χR) with next reshaped to (χR, d_next×χR_next) → (r, d_next×χR_next)
        # Then reshape back to (r, d_next, χR_next) for the next iteration.
        carry = reshape(SV * reshape(next, χR, d_next * χR_next), r, d_next, χR_next)
        χL = r
    end

    # Last site: carry is already the final tensor, just wrap it as a QTensor.
    tensors[L] = QTensor(
        carry, (upper(:vL, χL), upper(:σ, size(carry, 2)), lower(:vR, size(carry, 3)))
    )
    # `size(carry, 2)` = d_L (physical dim of last site), `size(carry, 3)` = 1 (right boundary).

    return FiniteMPS(tensors, bond_svs, CanonicalForm(L, L + 1), ε_total)
end

"""
    Base.:+(ψ::FiniteMPS, φ::FiniteMPS) -> FiniteMPS

Compute ``|\\psi\\rangle + |\\varphi\\rangle`` with unit coefficients.

Sugar for `add_mps(1, ψ, 1, φ)`.
"""
# `Base.:+` extends Julia's built-in + operator to work on FiniteMPS.
# `Base` is Julia's standard library module. `Base.:+` = the `+` function inside it.
# This is like Python's `def __add__(self, other):` but defined OUTSIDE the struct
# (Julia allows extending operators for any type, anywhere — open type system).
Base.:+(ψ::FiniteMPS, φ::FiniteMPS) = add_mps(1, ψ, 1, φ)
# One-liner function syntax: `f(x) = expr` is equivalent to `function f(x); return expr; end`

"""
    _scale(mps::FiniteMPS, a) -> FiniteMPS

Scale the MPS by a scalar `a`, absorbing `a` into the first site tensor.

Returns a new `FiniteMPS` with `form = ArbitraryForm()` and the same bond
structure, since scaling a single tensor generally destroys any isometry.
"""
function _scale(mps::FiniteMPS, a::Number)
    T = promote_type(typeof(a), eltype(mps.tensors[1].data))
    # Find common numeric type between the scalar `a` and the array element type.

    new_tensors = copy(mps.tensors)
    # `copy` creates a SHALLOW copy of the Vector — we get new vector but the QTensors inside
    # are still the same objects. We then replace tensors[1] below without modifying the original.
    # Like Python's `new_tensors = mps.tensors[:]` (shallow list copy).

    old_data = mps.tensors[1].data
    new_data = T.(a .* old_data)
    # `a .* old_data` = elementwise multiply: scales every element by `a` (like numpy: a * old_data).
    # `T.(...)` broadcasts the type conversion: equivalent to `convert(Array{T}, a .* old_data)`.
    # The `.` in `T.(...)` is Julia's broadcast syntax — applies T() to every element.

    new_tensors[1] = QTensor(new_data, mps.tensors[1].indices)
    # Replace the first tensor with the scaled version, keeping the same index metadata.

    return FiniteMPS(new_tensors, copy(mps.bond_svs), ArbitraryForm(), mps.ε)
    # Return ArbitraryForm because scaling breaks the normalization/isometry of the first tensor.
    # `copy(mps.bond_svs)` = shallow copy of the bond spectra vector (bond SVs are unchanged by scaling).
end
