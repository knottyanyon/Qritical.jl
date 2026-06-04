using LinearAlgebra: svd, norm, I, Diagonal

# ── AbstractMPS ───────────────────────────────────────────────────────────────

abstract type AbstractMPS end

# ── AbstractMPSForm hierarchy ─────────────────────────────────────────────────

"""
    AbstractMPSForm

Tag type recording the current gauge structure of a `FiniteMPS`.
Concrete subtypes are dispatchable so algorithm preconditions can be stated
in method signatures rather than checked at runtime.
"""
abstract type AbstractMPSForm end

"""
    CanonicalForm(llim, rlim)

Mixed canonical gauge.  `tensors[1:llim]` are left-isometric
(``A_i^\\dagger A_i = \\mathbb{1}``); `tensors[rlim:L]` are right-isometric
(``B_i B_i^\\dagger = \\mathbb{1}``).  The orthogonality center occupies
sites `llim+1` through `rlim-1`.

After a full left sweep: `CanonicalForm(L, L+1)`.
After a full right sweep: `CanonicalForm(0, 1)`.
"""
struct CanonicalForm <: AbstractMPSForm
    llim::Int
    rlim::Int
end

"""
    VidalForm()

Vidal ``\\Gamma``-``\\Lambda``-``\\Gamma`` gauge.  Site tensors in `tensors[i]`
are ``\\Gamma`` matrices; the ``\\Lambda`` (Schmidt value) arrays live in
`bond_svs`.  Natural form for TEBD.
"""
struct VidalForm <: AbstractMPSForm end

"""
    ArbitraryForm()

No orthogonality guarantees.  Initial form of any freshly-allocated `FiniteMPS`.
"""
struct ArbitraryForm <: AbstractMPSForm end

Base.:(==)(a::CanonicalForm,  b::CanonicalForm)  = a.llim == b.llim && a.rlim == b.rlim
Base.:(==)(::VidalForm,       ::VidalForm)        = true
Base.:(==)(::ArbitraryForm,   ::ArbitraryForm)    = true
Base.:(==)(::AbstractMPSForm, ::AbstractMPSForm)  = false

# ── FiniteMPS ─────────────────────────────────────────────────────────────────

"""
    FiniteMPS{D <: AbstractDoF, T <: Number, RT <: Real}

Finite matrix product state on `L` sites with physical degree of freedom `D`,
scalar type `T`, and real type `RT = real(T)`.

Each site tensor `tensors[i]` has three legs `(vL, σ, vR)`:
- `vL`: left virtual bond (`BondIndex`)
- `σ`:  physical index (`TIx{Lower}`, dimension `hilbert_space(D())`)
- `vR`: right virtual bond (`BondIndex`)

`bond_svs[i]` stores the Schmidt values at bond `i` (between sites `i-1`
and `i`).  `bond_svs[1] == bond_svs[L+1] == [1.0]` are the trivial open
boundaries.  Interior entries are populated during canonical sweeps.
"""
mutable struct FiniteMPS{D <: AbstractDoF, T <: Number, RT <: Real} <: AbstractMPS
    L         :: Int
    tensors   :: Vector{IndexedTensor{T, 3}}
    bond_svs  :: Vector{Vector{RT}}
    form      :: AbstractMPSForm
end

# ── constructor ───────────────────────────────────────────────────────────────

"""
    FiniteMPS(dof, L, χ; T=Float64)

Construct a random `FiniteMPS` with `L` sites, maximum bond dimension `χ`,
and scalar type `T`.  The actual bond dimension at bond `i` is
``\\min(d^{i-1}, d^{L-i+1}, \\chi)`` where ``d = \\operatorname{hilbert\\_space}(dof)``.
The form is `ArbitraryForm()`.
"""
function FiniteMPS(dof::D, L::Int, χ::Int; T::Type{<:Number}=Float64) where {D<:AbstractDoF}
    d  = hilbert_space(dof)
    RT = real(T)

    bond_dims = Vector{Int}(undef, L + 1)
    bond_dims[1]     = 1
    bond_dims[L + 1] = 1
    for i in 2:L
        bond_dims[i] = min(d^(i - 1), d^(L - i + 1), χ)
    end

    tensors = Vector{IndexedTensor{T, 3}}(undef, L)
    for i in 1:L
        χL = bond_dims[i]
        χR = bond_dims[i + 1]
        data = randn(T, χL, d, χR)
        vL = upper(bond_label(:χ, i - 1), χL)
        σ  = lower(:σ, d)
        vR = lower(bond_label(:χ, i),     χR)
        tensors[i] = IndexedTensor(data, (vL, σ, vR))
    end

    bond_svs         = Vector{Vector{RT}}(undef, L + 1)
    bond_svs[1]      = RT[1]
    bond_svs[L + 1]  = RT[1]
    for i in 2:L
        bond_svs[i] = fill(RT(NaN), bond_dims[i])
    end

    return FiniteMPS{D, T, RT}(L, tensors, bond_svs, ArbitraryForm())
end

# ── _update_tensor: reattach indices after shape change ───────────────────────

function _set_tensor!(mps::FiniteMPS, i::Int, data::AbstractArray{<:Number,3})
    old = mps.tensors[i]
    χL  = size(data, 1)
    χR  = size(data, 3)
    vL  = upper(bond_label(:χ, i - 1), χL)
    σ   = old.indices[2]
    vR  = lower(bond_label(:χ, i),     χR)
    mps.tensors[i] = IndexedTensor(data, (vL, σ, vR))
end

# ── left_canonical_sweep! ─────────────────────────────────────────────────────

"""
    left_canonical_sweep!(mps; trunc=KeepMachineEps())

Left-canonicalize `mps` site by site from left to right.  After the sweep
every tensor satisfies ``A_i^\\dagger A_i = \\mathbb{1}`` and the state is
normalized (``\\langle\\psi|\\psi\\rangle = 1``).  Interior `bond_svs` entries
are updated with the singular values at each bond.

An optional `trunc::AbstractTruncation` keyword truncates each bond.
"""
function left_canonical_sweep!(
    mps::FiniteMPS{D, T, RT};
    trunc::AbstractTruncation=KeepMachineEps(),
) where {D, T, RT}
    L = mps.L

    for i in 1:L
        data       = mps.tensors[i].data
        χL, d, χR = size(data)
        M          = reshape(data, χL * d, χR)
        F          = svd(M; full=false)

        r, _  = _truncate(F.S, trunc)
        r     = max(r, 1)

        A_data = reshape(F.U[:, 1:r], χL, d, r)
        _set_tensor!(mps, i, A_data)

        if i < L
            mps.bond_svs[i + 1] = copy(F.S[1:r])
            R        = Diagonal(F.S[1:r]) * F.Vt[1:r, :]  # (r, χR)
            next     = mps.tensors[i + 1].data              # (χR, d, χR')
            d_n, χRn = size(next, 2), size(next, 3)
            new_next = reshape(R * reshape(next, χR, d_n * χRn), r, d_n, χRn)
            _set_tensor!(mps, i + 1, new_next)
        end
    end

    mps.bond_svs[L + 1] = RT[1]
    mps.form = CanonicalForm(L, L + 1)
    return mps
end

# ── right_canonical_sweep! ────────────────────────────────────────────────────

"""
    right_canonical_sweep!(mps; trunc=KeepMachineEps())

Right-canonicalize `mps` from right to left.  After the sweep every tensor
satisfies ``B_i B_i^\\dagger = \\mathbb{1}`` and the state is normalized.
"""
function right_canonical_sweep!(
    mps::FiniteMPS{D, T, RT};
    trunc::AbstractTruncation=KeepMachineEps(),
) where {D, T, RT}
    L = mps.L

    for i in L:-1:1
        data       = mps.tensors[i].data
        χL, d, χR = size(data)
        M          = reshape(data, χL, d * χR)
        F          = svd(M; full=false)

        r, _ = _truncate(F.S, trunc)
        r    = max(r, 1)

        B_data = reshape(F.Vt[1:r, :], r, d, χR)
        _set_tensor!(mps, i, B_data)

        if i > 1
            mps.bond_svs[i] = copy(F.S[1:r])
            L_fac  = F.U[:, 1:r] * Diagonal(F.S[1:r])  # (χL, r)
            prev   = mps.tensors[i - 1].data              # (χL', d', χL)
            d_p, χL_new = size(prev, 2), size(prev, 1)
            new_prev = reshape(reshape(prev, χL_new * d_p, χL) * L_fac, χL_new, d_p, r)
            _set_tensor!(mps, i - 1, new_prev)
        end
    end

    mps.bond_svs[1] = RT[1]
    mps.form = CanonicalForm(0, 1)
    return mps
end

# ── move_center! ──────────────────────────────────────────────────────────────

"""
    move_center!(mps, target)

Move the orthogonality center of a canonically-gauged `mps` to site `target`,
updating `form` accordingly.  Assumes `mps.form isa CanonicalForm`.
"""
function move_center!(mps::FiniteMPS, target::Int)
    mps.form isa CanonicalForm || throw(ArgumentError(
        "move_center! requires CanonicalForm; got $(typeof(mps.form))"
    ))
    cf   = mps.form
    llim = cf.llim
    rlim = cf.rlim

    # Right-canonicalize sites from rlim-1 down to target+1:
    # each step makes one more site right-canonical and pushes the
    # factor leftward toward `target`.
    for i in (rlim - 1):-1:(target + 1)
        _svd_push_left!(mps, i)
    end

    # Left-canonicalize sites from llim+1 up to target-1:
    # each step makes one more site left-canonical and pushes the
    # factor rightward toward `target`.
    for i in (llim + 1):(target - 1)
        _svd_push_right!(mps, i)
    end

    # After both sweeps: sites 1..target-1 are left-canonical,
    # sites target+1..L are right-canonical, site `target` is the center.
    mps.form = CanonicalForm(target - 1, target + 1)
    return mps
end

# push tensor i left-canonical and absorb into i+1
function _svd_push_right!(mps::FiniteMPS, i::Int)
    data       = mps.tensors[i].data
    χL, d, χR = size(data)
    M          = reshape(data, χL * d, χR)
    F          = svd(M; full=false)
    r, _       = _truncate(F.S, KeepMachineEps())
    r          = max(r, 1)

    _set_tensor!(mps, i, reshape(F.U[:, 1:r], χL, d, r))
    mps.bond_svs[i + 1] = copy(F.S[1:r])

    if i < mps.L
        R    = Diagonal(F.S[1:r]) * F.Vt[1:r, :]
        next = mps.tensors[i + 1].data
        d_n  = size(next, 2)
        χRn  = size(next, 3)
        _set_tensor!(mps, i + 1, reshape(R * reshape(next, χR, d_n * χRn), r, d_n, χRn))
    end
end

# push tensor i right-canonical and absorb into i-1
function _svd_push_left!(mps::FiniteMPS, i::Int)
    data       = mps.tensors[i].data
    χL, d, χR = size(data)
    M          = reshape(data, χL, d * χR)
    F          = svd(M; full=false)
    r, _       = _truncate(F.S, KeepMachineEps())
    r          = max(r, 1)

    _set_tensor!(mps, i, reshape(F.Vt[1:r, :], r, d, χR))
    mps.bond_svs[i] = copy(F.S[1:r])

    if i > 1
        L_fac = F.U[:, 1:r] * Diagonal(F.S[1:r])
        prev  = mps.tensors[i - 1].data
        d_p   = size(prev, 2)
        χLp   = size(prev, 1)
        _set_tensor!(mps, i - 1, reshape(reshape(prev, χLp * d_p, χL) * L_fac, χLp, d_p, r))
    end
end

# ── overlap ───────────────────────────────────────────────────────────────────

"""
    overlap(bra, ket) -> scalar

Compute ``\\langle\\text{bra}|\\text{ket}\\rangle`` by boundary-to-boundary
contraction.  Works for any pair of `FiniteMPS` with the same `L` and physical
dimension.
"""
function overlap(bra::FiniteMPS, ket::FiniteMPS)
    bra.L == ket.L || throw(ArgumentError("MPS lengths differ"))
    T   = promote_type(eltype(bra.tensors[1]), eltype(ket.tensors[1]))
    env = ones(T, 1, 1)  # (χ_bra, χ_ket)

    for i in 1:bra.L
        A = conj.(bra.tensors[i].data)   # (χL_bra, d, χR_bra)
        B = ket.tensors[i].data           # (χL_ket, d, χR_ket)

        χL_bra, d, χR_bra = size(A)
        χL_ket, _,  χR_ket = size(B)

        # F[(α_ket,σ), β_bra] = Σ_{α_bra} env[α_bra,α_ket] * conj(A)[α_bra,σ,β_bra]
        # Use plain transpose (not adjoint) — A is already conjugated above.
        F_mat = reshape(transpose(env) * reshape(A, χL_bra, d * χR_bra), χL_ket * d, χR_bra)
        env   = transpose(F_mat) * reshape(B, χL_ket * d, χR_ket)
    end

    return env[1, 1]
end

# ── entanglement_entropy ──────────────────────────────────────────────────────

"""
    entanglement_entropy(mps, bond) -> Real

Entanglement entropy at `bond` (index into `bond_svs`) computed from the
stored Schmidt values:

```math
S = -\\sum_i \\lambda_i^2 \\log \\lambda_i^2
```

where ``\\lambda_i = \\sigma_i / \\|\\boldsymbol{\\sigma}\\|`` are the
normalized Schmidt coefficients.
"""
function entanglement_entropy(mps::FiniteMPS, bond::Int)
    s   = mps.bond_svs[bond]
    nrm = norm(s)
    nrm > 0 || return zero(real(eltype(mps.tensors[1])))
    λ = s ./ nrm
    return -sum(x -> x^2 * log(x^2), λ)
end

# ── MPS addition ──────────────────────────────────────────────────────────────

"""
    a * ψ + b * φ  (via Base.:+)

Return a new `FiniteMPS` representing ``a|\\psi\\rangle + b|\\phi\\rangle`` via
block-diagonal tensor concatenation.  Bond dimension of the result is
``\\chi_\\psi + \\chi_\\phi`` at interior bonds (1 at boundaries).
"""
function Base.:+(ψ::FiniteMPS{D, T, RT}, φ::FiniteMPS{D, T, RT}) where {D, T, RT}
    ψ.L == φ.L || throw(ArgumentError("MPS lengths differ"))
    L = ψ.L

    tensors = Vector{IndexedTensor{T, 3}}(undef, L)

    for i in 1:L
        A = ψ.tensors[i].data
        B = φ.tensors[i].data
        χL_ψ, d, χR_ψ = size(A)
        χL_φ, _,  χR_φ = size(B)

        if i == 1
            new = cat(A, B; dims=3)               # (1, d, χR_ψ + χR_φ)
        elseif i == L
            new = cat(A, B; dims=1)               # (χL_ψ + χL_φ, d, 1)
        else
            χL  = χL_ψ + χL_φ
            χR  = χR_ψ + χR_φ
            new = zeros(T, χL, d, χR)
            new[1:χL_ψ, :, 1:χR_ψ]             = A
            new[(χL_ψ+1):end, :, (χR_ψ+1):end] = B
        end

        vL = upper(bond_label(:χ, i - 1), size(new, 1))
        σ  = ψ.tensors[i].indices[2]
        vR = lower(bond_label(:χ, i),     size(new, 3))
        tensors[i] = IndexedTensor(new, (vL, σ, vR))
    end

    svs      = Vector{Vector{RT}}(undef, L + 1)
    svs[1]   = RT[1]
    svs[L+1] = RT[1]
    for i in 2:L
        svs[i] = fill(RT(NaN), size(tensors[i].data, 1))
    end

    return FiniteMPS{D, T, RT}(L, tensors, svs, ArbitraryForm())
end

# scalar * MPS
Base.:*(a::Number, mps::FiniteMPS{D, T, RT}) where {D, T, RT} = _scale(mps, a)
Base.:*(mps::FiniteMPS{D, T, RT}, a::Number) where {D, T, RT} = _scale(mps, a)

function _scale(mps::FiniteMPS{D, T, RT}, a::Number) where {D, T, RT}
    tensors = copy(mps.tensors)
    data    = a .* mps.tensors[1].data
    vL = mps.tensors[1].indices[1]
    σ  = mps.tensors[1].indices[2]
    vR = mps.tensors[1].indices[3]
    tensors[1] = IndexedTensor(data, (vL, σ, vR))
    return FiniteMPS{D, T, RT}(mps.L, tensors, deepcopy(mps.bond_svs), mps.form)
end
