# Glossary

## A

### Area Law

The area-law scaling of entanglement entropy: for a subsystem of size $L$, the entanglement entropy scales as the boundary area rather than the volume. For 1D chains, this means entropy is *constant* with subsystem size — a key feature that makes MPS efficient for ground states and low-energy states.

### Adjoint

The conjugate transpose of a tensor or operator. For a tensor leg, the adjoint simultaneously conjugates the values and flips the leg direction (Upper ↔ Lower). Denoted ``A^\dagger``.

### Bipartition

A partition of legs into two disjoint groups (row and column legs) used for reshaping a tensor into a matrix before SVD. For states, the bipartition marks a Schmidt cut; for operators, the bipartition separates bra (Upper) and ket (Lower) legs.

## B

### Bond

A virtual index connecting adjacent MPS tensors. Bond dimension $\chi$ controls compression: small $\chi$ gives fast algorithms, large $\chi$ gives accuracy.

### Bond Dimension

The dimension of a virtual bond in an MPS. Determines both the accuracy and computational cost: larger bond dimension means more entanglement can be represented but higher memory and CPU cost.

### Block-Sparse

A tensor stored as a collection of independent blocks, each labeled by a symmetry sector (e.g., charge, spin). Contractions only couple within blocks, giving a computational speedup when sectors are not entangled.

## C

### Canonical Form

An MPS gauge choice where some tensors satisfy left-isometry ($A^\dagger A = I$) or right-isometry ($B B^\dagger = I$). Canonical forms make expectation values computable in $O(L)$ time. See also: **Left-Canonical**, **Right-Canonical**, **Bond-Canonical**, **Site-Canonical**.

### Canonical Relation

A type tag specifying the algebra satisfied by creation/annihilation operators for a degree of freedom. Values: **CCR** (bosons, spins), **CAR** (fermions). This determines Jordan-Wigner signs and commutation vs. anticommutation.

### CCR

Canonical Commutation Relation. Operators satisfy $[a, a^\dagger] = 1$ (bosons, spins). See: **Canonical Relation**.

### CAR

Canonical Anti-Commutation Relation. Operators satisfy $\{a, a^\dagger\} = 1$ (fermions). See: **Canonical Relation**.

### Schmidt Decomposition

For a bipartite state $|\psi\rangle \in \mathcal{H}_A \otimes \mathcal{H}_B$, the unique factorization $|\psi\rangle = \sum_i \sigma_i |u_i\rangle_A \otimes |v_i\rangle_B$ where $\sigma_i \geq 0$ are singular values and $\{|u_i\rangle\}, \{|v_i\rangle\}$ are orthonormal. The Schmidt decomposition is an SVD of the reshaped state tensor.

### Schmidt Rank

The number of nonzero singular values in the Schmidt decomposition. A measure of bipartite entanglement: rank 1 means a product state; large rank means high entanglement.

### Schmidt Values

The singular values $\sigma_i$ in the Schmidt decomposition. Their squares $\sigma_i^2$ sum to 1 and represent the entanglement spectrum.

### Singular Value Decomposition (SVD)

Matrix factorization $A = U \Sigma V^\dagger$ where $U, V$ are unitary and $\Sigma$ is diagonal with non-negative entries. For tensors reshaped as matrices, SVD gives the Schmidt decomposition across the bipartition cut.

### Spectrum

In this library: a first-class object holding singular values (from SVD), eigenvalues (from diagonalization), or their physical interpretation as the entanglement spectrum. See: **SingValSpectrum**, **SchmidtSpectrum**, **EigValSpectrum**.

## D

### Degree of Freedom (DoF)

An elementary quantum system on a single site: spin-½, spinless fermion, hardcore boson, etc. DoFs carry information about local dimension, commutation relations (CCR/CAR), and physical operators (Pauli matrices, ladder operators, etc.).

## E

### Entanglement Entropy

The von Neumann entropy of the reduced density matrix across a bipartition: $S = -\sum_i \sigma_i^2 \log \sigma_i^2$ where $\sigma_i$ are Schmidt values. Measures bipartite quantum entanglement; ranges from 0 (product state) to $\log(\min(d^L, d^R))$ (maximal entanglement).

### Entanglement Spectrum

The negative logarithm of the Schmidt values: $\epsilon_i = -\log \sigma_i^2$. Captures fine structure of entanglement beyond entropy alone.

## G

### Gauge

A redundant degree of freedom in MPS that does not affect the state. Right-multiplying a site tensor by $X$ and left-multiplying the next by $X^{-1}$ leaves the state invariant. Gauging moves the orthogonality centre and controls numerical stability.

### Glossary

This document. Definitions of key terms used throughout Qritical.jl documentation and physics.

## I

### Index Variance

A label (Upper or Lower) on a tensor leg recording whether the leg is contravariant (domain, inward) or covariant (codomain, outward) in the sense of the von Delft arrow convention. Variance is not a convention — it follows directly from tensor-network geometry.

### Isometry

A unitary map between vector spaces. A left-isometric tensor $A$ satisfies $A^\dagger A = I$ (when reshaped as a matrix). A right-isometric tensor $B$ satisfies $B B^\dagger = I$.

## M

### Many-Body Localization (MBL)

A phase of matter at strong disorder where all eigenstates (not just the ground state) are localized: entanglement entropy obeys an area law, local operators fail to thermalize, and dynamics are non-diffusive. Marks a transition from ergodic to many-body localized behavior.

### Matrix Product State (MPS)

A compressed representation of a quantum state as a product of rank-3 tensors: $|\psi\rangle = \sum_{\sigma_1,\ldots,\sigma_L} A^{\sigma_1} A^{\sigma_2} \cdots A^{\sigma_L} |\sigma_1 \cdots \sigma_L\rangle$. Bond dimensions $\chi$ control accuracy; ground states of local gapped Hamiltonians satisfy an area law and need $\chi \ll d^{L/2}$.

### Matrix Product Operator (MPO)

The operator analogue of an MPS: an $d^L \times d^L$ operator factored into $L$ local rank-4 tensors. Used to apply Hamiltonians to states and compute expectation values efficiently.

## O

### Orthogonality Centre

The tensor in an MPS where the orthogonality property is concentrated. For left-canonical form, the centre is at the right boundary; for right-canonical, at the left; for bond-canonical, at a chosen bond; for site-canonical, at a chosen site. Re-gauging moves the centre without changing the state.

## P

### Partition

An ordered list of leg indices used for reshaping a tensor into a matrix. A bipartition pairs a row-partition with a column-partition for SVD.

### Product State

A state that factorizes across all sites: $|\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes \cdots \otimes |\psi_L\rangle$. Zero entanglement everywhere; MPS representation has $\chi = 1$.

## S

### Schmidt Cut

A bipartition of a state into two regions A and B. The Schmidt decomposition across this cut quantifies entanglement between the regions.

### Spectral Gap

The energy difference between the ground state and first excited state: $\Delta = E_1 - E_0$. Large gap means ground state is well-separated from excitations; small gap signals a phase transition.

### State Vector

A rank-$L$ tensor $|\psi\rangle_{(\sigma_1, \ldots, \sigma_L)}$ representing a pure quantum state. Has $d^L$ components; dense representation is exponential in system size.

## T

### TEBD (Time-Evolving Block Decimation)

An algorithm for time evolution of an MPS under a nearest-neighbour Hamiltonian. Uses Suzuki-Trotter product formula to factor the evolution operator into two-site gates applied sequentially, with truncation at each gate.

### Tensor Network

A graphical representation of a large tensor as a contraction of many small tensors. Edges represent indices, vertices represent tensors. Evaluating a network means contracting indices in some order.

### Truncation

Discarding small singular values in an SVD to keep only the top $r$ values. Reduces bond dimension and introduces controlled error $\varepsilon$ (the Frobenius norm of discarded weight). Essential for MPS efficiency.

### Truncation Error

The accumulated weight of singular values discarded during SVD: $\varepsilon = \|A - U \Sigma_{\rm trunc} V^\dagger\|_F$. Measured in Frobenius norm and affects state fidelity.

## U

### Upper/Lower Variance

Index direction labels in the von Delft arrow convention. **Upper** (contravariant) means the leg points inward/domain; **Lower** (covariant) means outward/codomain. Variance determines which leg indices can be contracted together in Einstein notation.

## V

### Virtual Bond

See: **Bond**.

### Von Delft Arrow Convention

A systematic labeling of tensor legs by *direction* (inward/outward) that makes contraction rules unambiguous without needing position-based index ordering. Used throughout Qritical.jl to avoid errors in multi-leg tensor operations.

---

**Last updated:** 2026-07-04
