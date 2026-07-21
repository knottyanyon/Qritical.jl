# ==== Bond ====================================================================

"""
    Bond

Pure geometry of one link in a tensor network: the pair of legs (faces) that
sit on either side of a shared index.

```math
\\cdots \\underbrace{-[\\,U\\,]}_{}\\!\\underbrace{-\\!\\overbrace{|\\lambda_L\\rangle\\langle\\lambda_R|}^{\\text{Bond}}\\!-}_{}[\\,V^\\dagger\\,]-\\cdots
```

A `Bond` carries no numerical data — it only records which `TIx` objects name
the two faces of the gauge centre ``\\Sigma``:

- `.left  :: TIx{Upper}` — ``\\Sigma``'s left face (``\\lambda_L``); its partner
  on ``U`` is the `Lower` leg of the same label.
- `.right :: TIx{Upper}` — ``\\Sigma``'s right face (``\\lambda_R``); its partner
  on ``V^\\dagger`` is the `Lower` leg of the same label.

Both faces are `Upper` because ``\\Sigma`` **is** the orthogonality centre: bond
arrows always point toward the centre, so both arrows point *into* ``\\Sigma``
(`Upper` = incoming = domain). The isometries on either side carry the matching
outgoing (`Lower`) ends, preserving the one-up-one-down contraction rule.

# Construction

After `do_svd`, the bond is derived directly from the ``\\Sigma`` factor:

```julia
F = do_svd(A, bp, NoTrunc())
F.center.bond.left  === F.Σ.indices[1]  # upper(:λL, r)
F.center.bond.right === F.Σ.indices[2]  # upper(:λR, r)
```

No label matching is needed — the legs are the *same* `TIx` objects.

# See also
[`BondCenter`](@ref), [`OrthoCenter`](@ref)
"""
struct Bond
    left::TIx{Upper}
    right::TIx{Upper}
end
