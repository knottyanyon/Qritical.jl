# §7.1–7.2  TEBD: time-evolution by block exponentiation + Trotter decomposition.
#
# A TEBD step applies a sequence of two-site gates drawn from the Suzuki-Trotter
# decomposition of e^{-iΔtH} (real time) or e^{-ΔτH} (imaginary time).
# For a nearest-neighbour chain the even/odd bond groups commute within each group,
# so each single-bond exponential is exact; only inter-group commutators contribute
# Trotter error.

# ─────────────────────────────────────────────────────────────────────────────
# Time axis types
# ─────────────────────────────────────────────────────────────────────────────

"""
    TimeAxis

Abstract supertype for the axis of time evolution.

Two concrete singletons:
- [`RealTime`](@ref)       — unitary evolution ``e^{-i \\Delta t H}``.
- [`ImaginaryTime`](@ref)  — non-unitary, PSD evolution ``e^{-\\Delta\\tau H}``;
  requires explicit renormalization after each step.
"""
abstract type TimeAxis end

"""
    RealTime <: TimeAxis

Selects unitary evolution ``e^{-i \\Delta t H}``.  A gate built with this axis
satisfies ``G^\\dagger G = I`` and preserves the MPS norm exactly.
"""
struct RealTime       <: TimeAxis end

"""
    ImaginaryTime <: TimeAxis

Selects imaginary-time evolution ``e^{-\\Delta\\tau H}``.  A gate built with this
axis is Hermitian and positive semidefinite but not unitary; the evolved state
must be renormalized at each step to keep it normalized.
"""
struct ImaginaryTime  <: TimeAxis end

# ─────────────────────────────────────────────────────────────────────────────
# Operator class tags
# ─────────────────────────────────────────────────────────────────────────────

"""
    Unitary

Tag returned by `opclass` for a real-time `Propagator`.  Indicates ``G^\\dagger G = I``.
"""
struct Unitary      end

"""
    HermitianPSD

Tag returned by `opclass` for an imaginary-time `Propagator`.
Indicates the gate is Hermitian and positive semidefinite.
"""
struct HermitianPSD end

# ─────────────────────────────────────────────────────────────────────────────
# Propagator — the gate tensor plus metadata
# ─────────────────────────────────────────────────────────────────────────────

"""
    Propagator{A<:TimeAxis}

A two-site evolution gate carrying its time axis `A` and time step `dt`.

Fields:
- `data::Matrix{ComplexF64}` — the ``d^2 \\times d^2`` gate matrix.
- `axis::A`                  — `RealTime()` or `ImaginaryTime()`.
- `dt::Float64`              — the time step used to build the gate.

Construct via [`gate`](@ref).  Query the physical class via [`opclass`](@ref).
"""
struct Propagator{A<:TimeAxis}
    data::Matrix{ComplexF64}
    axis::A
    dt::Float64
end

"""
    opclass(G::Propagator{RealTime})    -> Unitary()
    opclass(G::Propagator{ImaginaryTime}) -> HermitianPSD()

Return the algebraic class of the gate, derived purely from the time axis type.
"""
opclass(::Propagator{RealTime})      = Unitary()
opclass(::Propagator{ImaginaryTime}) = HermitianPSD()

# ─────────────────────────────────────────────────────────────────────────────
# Gate construction from a bond Hamiltonian
# ─────────────────────────────────────────────────────────────────────────────

"""
    gate(h::AbstractMatrix, dt::Real, ::RealTime)    -> Propagator{RealTime}
    gate(h::AbstractMatrix, dt::Real, ::ImaginaryTime) -> Propagator{ImaginaryTime}

Exponentiate a bond Hamiltonian `h` to produce a two-site gate.

For real time the gate is ``G = e^{-i \\Delta t h}``; for imaginary time
``G = e^{-\\Delta\\tau h}``.  The matrix exponential is evaluated by diagonalising
`h` (which must be Hermitian) and applying the scalar exponential to eigenvalues.
"""
function gate(h::AbstractMatrix, dt::Real, axis::RealTime)
    phase = -im * dt
    F = eigen(Hermitian(ComplexF64.(h)))
    data = F.vectors * Diagonal(exp.(phase .* F.values)) * F.vectors'
    Propagator{RealTime}(data, axis, Float64(dt))
end

function gate(h::AbstractMatrix, dt::Real, axis::ImaginaryTime)
    F = eigen(Hermitian(ComplexF64.(h)))
    data = F.vectors * Diagonal(exp.(-dt .* F.values)) * F.vectors'
    Propagator{ImaginaryTime}(data, axis, Float64(dt))
end

# ─────────────────────────────────────────────────────────────────────────────
# ConstantProtocol — a fixed-Hamiltonian evolution schedule
# ─────────────────────────────────────────────────────────────────────────────

"""
    ConstantProtocol{A<:TimeAxis}

An evolution protocol with a constant Hamiltonian, fixed time step, and fixed number
of steps.

Fields:
- `axis::A`         — `RealTime()` or `ImaginaryTime()`.
- `dt::Float64`     — time step per Trotter step.
- `nsteps::Int`     — total number of Trotter steps.
- `hamiltonian`     — the `Operator` driving the evolution.
"""
struct ConstantProtocol{A<:TimeAxis}
    axis::A
    dt::Float64
    nsteps::Int
    hamiltonian::Any   # Operator — typed as Any to avoid circular dependency
end

"""
    total_time(p::ConstantProtocol) -> Float64

Return the total evolution time ``\\Delta t \\times n_{\\text{steps}}``.
"""
total_time(p::ConstantProtocol) = p.dt * p.nsteps

"""
    gate(h::AbstractMatrix, p::ConstantProtocol{A}) -> Propagator{A}

Build a gate from bond Hamiltonian `h` and the protocol's time step and axis.
The gate axis matches the protocol axis, so the axis type is **never lost**.
"""
gate(h::AbstractMatrix, p::ConstantProtocol{A}) where {A} = gate(h, p.dt, p.axis)

# ─────────────────────────────────────────────────────────────────────────────
# Bond Hamiltonian extraction from an Operator
# ─────────────────────────────────────────────────────────────────────────────

"""
    bond_hamiltonian(H::Operator, b::Int) -> Matrix{ComplexF64}

Extract the ``d^2 \\times d^2`` two-site Hamiltonian for the ``b``-th bond of `H`.

The bond Hamiltonian is the sum of all bond terms acting on that bond pair
plus the **on-site fields split evenly** across its two endpoints:

```math
h_{ij} = \\sum_{\\text{BondTerms on }(i,j)} J_{ij}\\, O_i \\otimes O_j
        - \\frac{1}{2} h_i\\, S^z_i \\otimes I_j
        - \\frac{1}{2} h_j\\, I_i \\otimes S^z_j
```

Boundary sites appear in only one bond, so their full on-site contribution is
assigned to that bond (split-half accounting restores the correct total).
"""
function bond_hamiltonian(H::Operator, b::Int)
    d   = local_dim(H.dof)
    Id  = Matrix{ComplexF64}(I, d, d)
    bnd = bonds(H.geom)
    i, j = bnd[b]
    L = H.geom.L

    h = zeros(ComplexF64, d^2, d^2)

    # Bond operator contributions
    for bt in H.bond
        (bt.i == i && bt.j == j) || continue
        h .+= bt.coupling .* kron(ComplexF64.(bt.op_i), ComplexF64.(bt.op_j))
    end

    # On-site contributions: split each site's weight between its neighbouring bonds
    for lt in H.onsite
        # Count how many bonds site lt.site participates in
        n_bonds = count(bnd_pair -> lt.site ∈ bnd_pair, bnd)
        # Weight = 1/n_bonds for each bond
        weight  = lt.coupling / n_bonds
        if lt.site == i
            h .+= weight .* kron(ComplexF64.(lt.op), Id)
        elseif lt.site == j
            h .+= weight .* kron(Id, ComplexF64.(lt.op))
        end
    end

    return h
end

# ─────────────────────────────────────────────────────────────────────────────
# Two-site gate application
# ─────────────────────────────────────────────────────────────────────────────

"""
    apply_gate(ψ::FiniteMPS, G::Propagator, bond::Int; trunc) -> FiniteMPS

Apply a two-site gate `G` at bond `bond` (connecting sites `bond` and `bond+1`).

Steps:
1. Merge the two MPS tensors: ``\\Theta[\\alpha, \\sigma_1, \\sigma_2, \\beta]``.
2. Contract with gate ``G[\\sigma_1', \\sigma_2', \\sigma_1, \\sigma_2]``.
3. Reshape and SVD-split with truncation `trunc`.
4. Store left-canonical ``A[\\alpha, \\sigma_1', r]`` and remainder ``B[r, \\sigma_2', \\beta]``.

For `RealTime` gates the norm of the MPS is preserved exactly (up to floating-point
error). For `ImaginaryTime` gates the norm decreases; the caller is responsible for
renormalization.
"""
function apply_gate(ψ::FiniteMPS, G::Propagator, bond::Int;
                    trunc::AbstractTrunc=NoTrunc())::FiniteMPS
    L  = length(ψ.tensors)
    i, j = bond, bond + 1
    (1 ≤ i < L) || throw(ArgumentError("bond $bond out of range [1, $(L-1)]"))

    A1 = ψ.tensors[i].data   # (χL, d, χM)
    A2 = ψ.tensors[j].data   # (χM, d, χR)
    χL, d, χM = size(A1)
    _,  _, χR = size(A2)

    # Merge: Θ[χL, d, d, χR]
    @tensor Θ[α, σ1, σ2, β] := A1[α, σ1, m] * A2[m, σ2, β]

    # Apply gate: G has shape (d², d²) with row=(σ1',σ2'), col=(σ1,σ2)
    G_mat = reshape(G.data, d, d, d, d)   # [σ1', σ2', σ1, σ2]
    @tensor Θ_new[α, σ1p, σ2p, β] := G_mat[σ1p, σ2p, σ1, σ2] * Θ[α, σ1, σ2, β]

    # SVD split: reshape to (χL*d, d*χR)
    M = reshape(permutedims(Θ_new, (1,2,3,4)), χL * d, d * χR)
    F = svd(M)
    tol     = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
    S_clean = filter(s -> s > tol, F.S)
    r, _    = _truncate_singular_values(S_clean, trunc)
    svs     = F.S[1:r]

    A1_new = reshape(F.U[:, 1:r], χL, d, r)           # left-canonical
    A2_new = reshape(Diagonal(svs) * F.Vt[1:r, :], r, d, χR)  # absorb Σ

    # Build new tensor list with the two updated sites
    tensors  = copy(ψ.tensors)
    bond_svs = copy(ψ.bond_svs)

    tensors[i]    = QTensor(A1_new, (upper(:vL, χL), lower(:σ, d), lower(:vR, r)))
    tensors[j]    = QTensor(A2_new, (upper(:vL, r),  lower(:σ, d), lower(:vR, χR)))
    normalized    = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
    bond_svs[i+1] = SingValSpectrum(svs, 0.0, normalized)

    FiniteMPS(tensors, bond_svs, ArbitraryForm(), ψ.ε)
end

# ─────────────────────────────────────────────────────────────────────────────
# Suzuki-Trotter decomposition
# ─────────────────────────────────────────────────────────────────────────────

"""
    TrotterSubstep

One substep in a Suzuki-Trotter decomposition: the index of the bond and the
time step to apply.
"""
struct TrotterSubstep
    bond::Int
    dt::Float64
end

"""
    SuzukiTrotter{order}

Suzuki-Trotter product formula of the given `order`.

- `SuzukiTrotter(1)` — first-order (Lie-Trotter): one sequential sweep through all bonds.
  Error ``O(\\Delta t^2)`` per step.
- `SuzukiTrotter(2)` — second-order (Strang / symmetric): symmetric palindrome.
  Error ``O(\\Delta t^3)`` per step.

The higher-order schemes achieve better accuracy at the same ``\\Delta t`` by choosing
carefully fractional sub-steps that cancel leading commutator error terms.
"""
struct SuzukiTrotter{order} end
SuzukiTrotter(order::Int) = SuzukiTrotter{order}()

"""
    trotter_steps(formula::SuzukiTrotter, H::Operator, dt::Real) -> Vector{TrotterSubstep}

Return the ordered list of single-bond gate applications for one Trotter step.

Each `TrotterSubstep` carries a bond index and the sub-step time `dt` to pass to
[`gate`](@ref).  The returned sequence, applied left-to-right, approximates
``e^{-i \\Delta t H}`` to the order of the formula.
"""
function trotter_steps(::SuzukiTrotter{1}, H::Operator, dt::Real)
    bnd = bonds(H.geom)
    [TrotterSubstep(b, Float64(dt)) for b in 1:length(bnd)]
end

function trotter_steps(::SuzukiTrotter{2}, H::Operator, dt::Real)
    bnd  = bonds(H.geom)
    n    = length(bnd)
    half = Float64(dt) / 2
    # Forward half-step for all but last bond, full step for last, then reverse
    fwd  = [TrotterSubstep(b, half) for b in 1:n]
    bwd  = [TrotterSubstep(b, half) for b in n:-1:1]
    # True 2nd-order: dt/2 each bond forward, then dt/2 each bond backward
    # (palindrome: same bond sequence in reverse at half dt)
    vcat(fwd, bwd)
end

# ─────────────────────────────────────────────────────────────────────────────
# Single Trotter step
# ─────────────────────────────────────────────────────────────────────────────

"""
    trotter_step(ψ::FiniteMPS, H::Operator, dt::Real,
                 formula::SuzukiTrotter; trunc) -> FiniteMPS

Apply one Trotter step of the evolution ``e^{-i \\Delta t H}`` to `ψ`.

The step decomposes `H` into bond sub-steps via [`trotter_steps`](@ref), computes
the gate for each bond, and applies them in sequence via [`apply_gate`](@ref).

For real-time evolution (`RealTime`) the norm is preserved.  For imaginary-time
evolution (`ImaginaryTime`) the caller should renormalize after the step.
"""
function trotter_step(ψ::FiniteMPS, H::Operator, dt::Real,
                      formula::SuzukiTrotter;
                      trunc::AbstractTrunc=NoTrunc(),
                      axis::TimeAxis=RealTime())

    substeps = trotter_steps(formula, H, dt)
    cur = ψ
    for sub in substeps
        h_b = bond_hamiltonian(H, sub.bond)
        G   = gate(h_b, sub.dt, axis)
        cur = apply_gate(cur, G, sub.bond; trunc=trunc)
    end
    cur
end
