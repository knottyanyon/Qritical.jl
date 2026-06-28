# Reading Notes: Kennes & Karrasch — *Extending the range of real time DMRG simulations*

**Reference:** D. M. Kennes and C. Karrasch, *Comp. Phys. Comm.* **200**, 37 (2016) [arXiv:1404.3704].
**Scope of these notes:** Abstract, §1 (Introduction), §2 (Schrödinger vs. Heisenberg picture) — i.e. up to but *not* including §3.

---

## 0. One-line takeaway

The paper collects a few cheap, "beginner-friendly" tricks that roughly **double the reachable simulation time** in time-dependent DMRG (tDMRG) on MPS, at *the same bond dimension* $\chi$ — exploiting the fact that the cost-limiting quantity, $\chi$, typically grows **exponentially** in time.

---

## 1. The core problem

A pure state is an MPS:

$$|\psi\rangle = \sum_{\{\sigma_l\}} M^{\sigma_1} M^{\sigma_2}\cdots M^{\sigma_L}\,|\{\sigma_l\}\rangle .$$

Thanks to the **area law**, ground states need only a modest bond dimension $\chi$. But for *dynamics* — correlation functions or quench dynamics — entanglement grows physically with time, so the $\chi$ needed for fixed accuracy **grows with $t$, often exponentially.** This is the wall that limits accessible time scales.

Two standard objects of interest:

| Quantity | Expression |
|---|---|
| Ground-state correlation function | $C^{AB}_{gs}(t) = \langle gs|A(t)B|gs\rangle$, with $A(t)=e^{iHt}A\,e^{-iHt}$ |
| Quench dynamics in arbitrary state | $\langle A\rangle_\psi(t) = \langle\psi|A(t)|\psi\rangle$ |

The paper's goal: assuming you already have a *standard MPS-based tDMRG code*, what minimal recipes extend its time reach?

---

## 2. The four tricks (overview)

The paper advertises four ideas. The portion we read (§1–§2) introduces all of them conceptually and works the first two:

1. **Combine Schrödinger & Heisenberg pictures** (§2) — split the evolution between state and operator.
2. **Compute the operator evolution $A(t)$ inside an MPS code** (§3 — *not yet read*).
3. **Exploit time-translation invariance** for a factor-of-2 in equilibrium correlators (§5).
4. **Analytic understanding of the finite-$T$ disentangler** (§4).

---

## 3. Trick A — "A factor of two" via time-translation invariance (introduced in §1)

For an *equilibrium* correlator, instead of the standard route (evolve $e^{-iHt}B|gs\rangle$), recast it symmetrically:

$$C^{AB}_{gs}(2t) = \langle gs|\,A(t)\,B(-t)\,|gs\rangle .$$

Run **two** separate simulations, $e^{-iHt}B|gs\rangle$ and $e^{+iHt}A|gs\rangle$, each only to time $t$. Stitching them gives the correlator out to $2t$ — **twice the time at no extra cost.** The authors note this simple point was long overlooked.

---

## 4. Trick B — Schrödinger vs. Heisenberg picture (§2, the main worked idea)

### The two pictures

To evaluate $\langle A\rangle_\psi(t)=\langle\psi|e^{iHt}A e^{-iHt}|\psi\rangle$:

| Picture | What you evolve | MPS/MPO object |
|---|---|---|
| **Schrödinger** | the state $e^{-iHt}|\psi\rangle$ | MPS |
| **Heisenberg** | the operator $A(t)=e^{iHt}A e^{-iHt}$ | MPO |

Mathematically equivalent, **algorithmically different**: one representation may stay compact while the other blows up.

### Two clean limiting cases (spin-½ XXZ chain)

$$H=\sum_{l=1}^{L-1}\Big[\tfrac12\big(S^+_l S^-_{l+1}+S^-_l S^+_{l+1}\big)+\Delta\,S^z_l S^z_{l+1}\Big]+b\sum_{l=1}^{L}S^z_l$$

- **$\Delta = 0$ (free/XX point):** $S^z_l(t)$ has an *exact finite-$\chi$ MPO*. So in the **Heisenberg** picture you can reach arbitrarily long times for *any* state $|\psi\rangle$.
- **$|\psi\rangle$ an eigenstate of $H$:** the **Schrödinger** picture is trivial (state barely evolves) for any $A$, any $\Delta$.

### The generic case → split the evolution

For a generic $\Delta = O(1)$, *both* representations grow equally fast. The trick: **split the time budget between the two pictures**:

$$\langle A\rangle_\psi(2t)=\langle\psi|\,e^{iHt}A(t)\,e^{-iHt}\,|\psi\rangle,$$

evolving $e^{-iHt}|\psi\rangle$ and $A(t)$ *separately*, each stopped when it hits the $\chi$ ceiling at times $t_\psi$ and $t_A$. Total reach $t_\psi+t_A > $ either alone. When the two problems are "equally complex" ($t_\psi \approx t_A$), you gain roughly a **factor of 2**.

The finite-temperature analogue (with initial density matrix $\rho$):

$$\langle A\rangle_\rho(2t)=\mathrm{Tr}\big[\rho(-t)\,A(t)\big].$$

### When the split is *not* worth it

The extra effort isn't justified when one side is trivial:
- $|\psi\rangle$ near an eigenstate of $H$ (or $\rho$ near thermal), **or**
- $A(t)$ has an efficient MPO (e.g. $A=S^z_l$, $\Delta=0$).

Then $t_\psi \ll t_A$ (or vice versa) and you gain little.

---

## 5. Numerical evidence (§2)

**Example 1 — global quench, $S^z_{L/2}(t)$ in a Néel state** (Fig. 1; XXZ, $\Delta=0.5$, $b=0$, $L=100$):
- Splitting Schrödinger+Heisenberg roughly **doubles the reachable time** at the same $\chi$.
- Dramatic stakes: reaching $t\sim 20$ purely in Schrödinger would need $\chi\sim 30000$ (by extrapolation), versus $\chi\sim 900$ when both pictures are combined. Because $\chi$ grows exponentially, the factor-of-2 in *time* is a **massive** saving in *cost*.

**Example 2 — local quench at finite $T$** (Fig. 2; isotropic $\Delta=1$, $L=100$):
- Initial state: a wave packet of four up-spins at the center, flanked by thermal chains, $\rho=\rho^L_T\otimes\rho_{\uparrow\uparrow\uparrow\uparrow}\otimes\rho^R_T$.
- Computed via $\langle S^z_l\rangle_\rho(t)=\mathrm{Tr}[\rho\,e^{iHt}S^z_l e^{-iHt}]=\langle\psi_\rho|e^{iHt}S^z_l e^{-iHt}|\psi_\rho\rangle$, where $\rho=\mathrm{Tr}_Q|\psi_\rho\rangle\langle\psi_\rho|$ (purification).
- A **disentangler** $e^{iH_Q t}$ acting on the auxiliary space is inserted to slow entanglement growth; combining pictures again reaches larger times. (The *why* of the disentangler is §4 — not yet read.)

---

## 6. Finite-temperature setup (introduced in §1, used in §2)

The thermal operator is written as a **purification** — a partial trace over a pure state in an enlarged Hilbert space with auxiliary ("bath") sites $Q$:

$$\rho_T = \mathrm{Tr}_Q\,|\Psi_T\rangle\langle\Psi_T| .$$

A finite-$T$ correlator follows from real- and imaginary-time evolution of the infinite-$T$ purification $|\psi_\infty\rangle$:

$$C^{AB}_T(t) = \langle\psi_T|A(t)B|\psi_T\rangle \sim \langle\psi_\infty|\,e^{-H/2T}A(t)B\,e^{-H/2T}\,|\psi_\infty\rangle .$$

**Key freedom (the disentangler idea):** purification is *not unique*. Any unitary $U_Q(t)$ acting only on the auxiliary space leaves $\rho_T$ invariant, so it can be inserted freely:

$$C^{AB}_T(t)=\langle\psi_T|\,U_Q^\dagger(t)\,A(t)\,U_Q(t)\,B\,|\psi_T\rangle .$$

The choice $U_Q(t)=e^{iH_Q t}$ (evolve the auxiliary space *backward* under the physical Hamiltonian) empirically **slows the bond-dimension growth** → longer times. Originally a "lucky guess"; §4 gives the analytic reason (sign-flip argument on $H_Q$ terms when symmetries are used).

The finite-$T$ factor-of-two analogue:

$$C^{AB}_T(2t)=\mathrm{Tr}\big[\rho_T A(t)B(-t)\big]=\langle\psi_T|A(t)B(-t)|\psi_T\rangle .$$

---

## 7. Practical / numerical conventions stated so far

- **4th-order Trotter decomposition** of the time-evolution operators.
- Truncation controlled by a **fixed small discarded weight**.
- The XXZ spin chain and the Hubbard model are the two running examples; Abelian symmetries (spin, charge) will be exploited.

---

## 8. Open threads to pick up after §3

- §3: *how* to actually time-evolve the operator $A(t)$ as an MPO inside a standard MPS code (the "trivial way" via $d^2$ superindex vs. the finite-$T$/SVD recasting); cost scalings $d^6\chi^3$ vs. $d^4\chi^3$.
- §4: analytic justification of the disentangler + the symmetry sign-flip rule.
- §5: the factor-of-two recast at $T>0$, $C^{AB}_T(2t)=\langle\psi_T|A\,e^{-iHt+iH_Q t}\,e^{-iHt+iH_Q t}B|\psi_T\rangle$, and the spin-current MPO example.

---

## Questions seeded for our Q&A session

(placeholders — to flesh out together after my own read)

1. Why does splitting the evolution help *only* when $t_\psi\approx t_A$ — what's the precise cost argument?
2. In the $\Delta=0$ XX case, why is the $S^z_l(t)$ MPO bond dimension finite/exact?
3. Physical intuition: why does evolving the auxiliary space *backward* in time disentangle?
4. How does the factor-of-two trick interact with the picture-splitting trick — are the gains multiplicative?
