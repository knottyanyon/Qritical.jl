# Kitchen Sink

Reference page for all custom doc components — math environments, admonition styles,
citation patterns, and proof fold syntax.  Not part of the public docs; kept here so
new components can be evaluated quickly against the full design system.

---

## Math environments

Six typed boxes rendered via `@raw html`.  `math-env.js` injects the auto-numbered
tab badge (top-right corner).  Each box uses the **split-block** pattern so that
body content is regular Documenter markdown — LaTeX, `@cite`, cross-references, and
everything else work normally.

**Usage skeleton:**

````markdown
```@raw html
<div class="math-env math-theorem" id="thm-my-label">
  <div class="math-env-header">Title with \(\LaTeX\)</div>
  <div class="math-env-body content">
```

Body text with $\LaTeX$ and [citations](@cite).

```@raw html
    <details class="math-proof-fold">
      <summary>Proof</summary>
      <div class="math-proof-body content">
```

Proof steps. $\square$

```@raw html
      </div>
    </details>
  </div>
</div>
```
````

Cross-page links: `[Theorem 1](path/to/page.html#thm-my-label)` — the `id` is a
standard HTML anchor so it works anywhere.

---

### Definition

```@raw html
<div class="math-env math-definition" id="ks-def-frobenius">
  <div class="math-env-header">Frobenius norm of \(A \in \mathbb{C}^{m \times n}\)</div>
  <div class="math-env-body content">
```

$$\|A\|_F \;=\; \left(\sum_{i,j} |a_{ij}|^2\right)^{1/2} \;=\; \sqrt{\operatorname{tr}(A^* A)}.$$

Equivalently the $\ell^2$-norm of $A$ flattened to a vector, or the $\ell^2$-norm of
its singular values [golub_vanloan_2013](@cite).

```@raw html
  </div>
</div>
```

### Theorem (with foldable proof and cross-reference)

```@raw html
<div class="math-env math-theorem" id="ks-thm-schmidt">
  <div class="math-env-header">Schmidt Decomposition of \(\ket{\psi}\)</div>
  <div class="math-env-body content">
```

For any $\ket{\psi} \in \mathcal{H}_A \otimes \mathcal{H}_B$ there exist orthonormal
sets $\{\ket{u_i}\} \subset \mathcal{H}_A$, $\{\ket{v_i}\} \subset \mathcal{H}_B$ and
strictly positive reals $\lambda_i$ with $\sum_i \lambda_i^2 = 1$ such that

$$\ket{\psi} = \sum_{i=1}^{r} \lambda_i \ket{u_i} \otimes \ket{v_i}.$$

The integer $r = \operatorname{Sch}(\psi)$ is the *Schmidt rank*
(see [Definition 1](#ks-def-frobenius) for the singular-value connection)
[trefethen_1997](@cite).

```@raw html
    <details class="math-proof-fold">
      <summary>Proof</summary>
      <div class="math-proof-body content">
```

Write $\ket{\psi} = \sum_{ij} C_{ij}\ket{i}\ket{j}$ and apply the SVD
$C = U\Sigma V^\dagger$.  Left/right singular vectors give $\ket{u_i}$, $\ket{v_i}$;
singular values give $\lambda_i$.  Unitarity of $U$, $V$ establishes orthonormality;
$\|\psi\|^2 = 1$ forces $\sum_i \lambda_i^2 = 1$. $\square$

```@raw html
      </div>
    </details>
  </div>
</div>
```

### Lemma

```@raw html
<div class="math-env math-lemma" id="ks-lem-schmidt-bound">
  <div class="math-env-header">\(\operatorname{Sch}(\psi) \leq \min(d_A, d_B)\)</div>
  <div class="math-env-body content">
```

The Schmidt rank is bounded by the smaller Hilbert-space dimension, since the
coefficient matrix $C \in \mathbb{C}^{d_A \times d_B}$ (see proof of
[Theorem 1](#ks-thm-schmidt)) has at most $\min(d_A, d_B)$ nonzero singular values.

```@raw html
  </div>
</div>
```

### Corollary

```@raw html
<div class="math-env math-corollary" id="ks-cor-eckart-young">
  <div class="math-env-header">Eckart&ndash;Young best rank-\(\nu\) approximation</div>
  <div class="math-env-body content">
```

The truncated SVD $A_\nu = \sum_{j=1}^\nu \sigma_j u_j v_j^*$ minimises $\|A - B\|_F$
over all rank-$\nu$ matrices $B$, where $\|\cdot\|_F$ is
[Definition 1](#ks-def-frobenius) [golub_vanloan_2013](@cite).

```@raw html
  </div>
</div>
```

### Proposition

```@raw html
<div class="math-env math-proposition" id="ks-prop-entanglement-entropy">
  <div class="math-env-header">Entanglement entropy of \(\ket{\psi}\)</div>
  <div class="math-env-body content">
```

With the Schmidt decomposition of [Theorem 1](#ks-thm-schmidt),

$$S(\psi) = -\sum_{i=1}^r \lambda_i^2 \log \lambda_i^2.$$

$S(\psi) = 0$ iff $\ket{\psi}$ is a product state
(i.e. [Lemma 1](#ks-lem-schmidt-bound) gives $r = 1$).

```@raw html
  </div>
</div>
```

### Remark (no title)

```@raw html
<div class="math-env math-remark">
  <div class="math-env-header"></div>
  <div class="math-env-body content">
```

A rapidly decaying spectrum $\lambda_1 \gg \lambda_2 \gg \cdots$ signals low
entanglement — exactly when MPS methods are efficient.

```@raw html
  </div>
</div>
```

---

## Admonition styles

!!! theorem "Norm identities"
    $$\|A\|_2 = \sigma_1, \qquad \|A\|_F = \sqrt{\sigma_1^2 + \cdots + \sigma_r^2}.$$

!!! definition "Induced matrix norm"
    The smallest $C$ such that $\|Ax\|_{(m)} \leq C\|x\|_{(n)}$ for all $x$.

!!! question "Exercise"
    Verify that the Frobenius norm is unitarily invariant.

!!! algorithm "Power iteration"
    Repeat $v \leftarrow Av / \|Av\|$ until convergence.

!!! info "Note"
    Standard Documenter info admonition.

!!! warning "Warning"
    Standard Documenter warning admonition.

---

## References

```@bibliography
Pages = ["kitchen_sink.md"]
Canonical = false
```
