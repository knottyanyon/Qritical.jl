```@meta
EditURL = "task_3.jl"
```

# Task 1.3 — SVD a state

!!! question "Task 1.3"
    Perform an SVD on the state `psi.jls` — a rank-10 tensor with shape
    ``(2, 2, \ldots, 2)`` representing a 10-qubit state. Find the Schmidt rank
    needed if singular values below ``10^{-6}`` are discarded for:
    - **(a)** a bipartition after the first site;
    - **(b)** a bipartition at the middle (5 | 5).

````julia
DATA_ROOT = normpath(joinpath(@__FILE__, "..", "..", "data"))
FPATH_PSI = joinpath(DATA_ROOT, "psi.jls")
````

````
"/Users/bavithra/Documents/Uni/Courses/26_HTN/Qritical.jl/docs/src/exercises/data/psi.jls"
````

````julia
using Serialization, LinearAlgebra, Qritical, CairoMakie

ψ = deserialize(FPATH_PSI)    # shape (2,2,...,2): 10 sites, each dim-2
N = ndims(ψ)
````

````
10
````

Attach a named upper index to each site leg.  All sites share dimension 2.
Using `Symbol(:s, i)` gives labels `:s1, :s2, …, :s10`.

````julia
sites = [upper(Symbol(:s, i), 2) for i in 1:N]
A     = IndexedTensor(ψ, Tuple(sites))
````

````
2×2×2×2×2×2×2×2×2×2 Qritical.IndexedTensor{Float64, 10, Array{Float64, 10}}:
[:, :, 1, 1, 1, 1, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 1, 1, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 1, 1, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 2, 1, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 2, 1, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 2, 1, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 2, 1, 1, 1, 1, 1] =
 0.0   0.0
 0.0  -4.45483e-5

[:, :, 1, 1, 1, 2, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 2, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 1, 2, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 1, 2, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.000268478

[:, :, 1, 1, 2, 2, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 2, 2, 1, 1, 1, 1] =
 0.0   0.0
 0.0  -0.00078674

[:, :, 1, 2, 2, 2, 1, 1, 1, 1] =
 0.0  0.0
 0.0  0.00144039

[:, :, 2, 2, 2, 2, 1, 1, 1, 1] =
  0.0         0.00121646
 -0.00173367  0.0

[:, :, 1, 1, 1, 1, 2, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 2, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 1, 1, 2, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 1, 1, 2, 1, 1, 1] =
 0.0   0.0
 0.0  -0.00078674

[:, :, 1, 1, 2, 1, 2, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 2, 1, 2, 1, 1, 1] =
 0.0  0.0
 0.0  0.00303256

[:, :, 1, 2, 2, 1, 2, 1, 1, 1] =
 0.0   0.0
 0.0  -0.00616033

[:, :, 2, 2, 2, 1, 2, 1, 1, 1] =
 0.0         -0.00559752
 0.00779143   0.0

[:, :, 1, 1, 1, 2, 2, 1, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 2, 2, 1, 1, 1] =
 0.0   0.0
 0.0  -0.00438211

[:, :, 1, 2, 1, 2, 2, 1, 1, 1] =
 0.0  0.0
 0.0  0.0116885

[:, :, 2, 2, 1, 2, 2, 1, 1, 1] =
  0.0        0.0122513
 -0.0163391  0.0

[:, :, 1, 1, 2, 2, 2, 1, 1, 1] =
 0.0   0.0
 0.0  -0.0107416

[:, :, 2, 1, 2, 2, 2, 1, 1, 1] =
 0.0        -0.0160458
 0.0195825   0.0

[:, :, 1, 2, 2, 2, 2, 1, 1, 1] =
  0.0       0.0127685
 -0.012182  0.0

[:, :, 2, 2, 2, 2, 2, 1, 1, 1] =
 -0.00508031  0.0
  0.0         0.0

[:, :, 1, 1, 1, 1, 1, 2, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 1, 2, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 1, 1, 1, 2, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 1, 1, 1, 2, 1, 1] =
 0.0  0.0
 0.0  0.00144039

[:, :, 1, 1, 2, 1, 1, 2, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 2, 1, 1, 2, 1, 1] =
 0.0   0.0
 0.0  -0.00616033

[:, :, 1, 2, 2, 1, 1, 2, 1, 1] =
 0.0  0.0
 0.0  0.0131734

[:, :, 2, 2, 2, 1, 1, 2, 1, 1] =
  0.0        0.0124752
 -0.0171258  0.0

[:, :, 1, 1, 1, 2, 1, 2, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 2, 1, 2, 1, 1] =
 0.0  0.0
 0.0  0.0116885

[:, :, 1, 2, 1, 2, 1, 2, 1, 1] =
 0.0   0.0
 0.0  -0.0328197

[:, :, 2, 2, 1, 2, 1, 2, 1, 1] =
 0.0        -0.0358522
 0.0471566   0.0

[:, :, 1, 1, 2, 2, 1, 2, 1, 1] =
 0.0  0.0
 0.0  0.0334649

[:, :, 2, 1, 2, 2, 1, 2, 1, 1] =
  0.0       0.0521005
 -0.062709  0.0

[:, :, 1, 2, 2, 2, 1, 2, 1, 1] =
 0.0        -0.0436437
 0.0410656   0.0

[:, :, 2, 2, 2, 2, 1, 2, 1, 1] =
 0.0178488  0.0
 0.0        0.0

[:, :, 1, 1, 1, 1, 2, 2, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 2, 2, 1, 1] =
 0.0   0.0
 0.0  -0.0107416

[:, :, 1, 2, 1, 1, 2, 2, 1, 1] =
 0.0  0.0
 0.0  0.0334649

[:, :, 2, 2, 1, 1, 2, 2, 1, 1] =
  0.0       0.039332
 -0.050527  0.0

[:, :, 1, 1, 2, 1, 2, 2, 1, 1] =
 0.0   0.0
 0.0  -0.0448849

[:, :, 2, 1, 2, 1, 2, 2, 1, 1] =
 0.0        -0.0751842
 0.0883824   0.0

[:, :, 1, 2, 2, 1, 2, 2, 1, 1] =
  0.0        0.06988
 -0.0642187  0.0

[:, :, 2, 2, 2, 1, 2, 2, 1, 1] =
 -0.0300308  0.0
  0.0        0.0

[:, :, 1, 1, 1, 2, 2, 2, 1, 1] =
 0.0  0.0
 0.0  0.0268122

[:, :, 2, 1, 1, 2, 2, 2, 1, 1] =
  0.0        0.0518072
 -0.0583517  0.0

[:, :, 1, 2, 1, 2, 2, 2, 1, 1] =
 0.0       -0.0632262
 0.055671   0.0

[:, :, 2, 2, 1, 2, 2, 2, 1, 1] =
 0.0300308  0.0
 0.0        0.0

[:, :, 1, 1, 2, 2, 2, 2, 1, 1] =
  0.0        0.0288143
 -0.0232168  0.0

[:, :, 2, 1, 2, 2, 2, 2, 1, 1] =
 -0.0178488  0.0
  0.0        0.0

[:, :, 1, 2, 2, 2, 2, 2, 1, 1] =
 0.00508031  0.0
 0.0         0.0

[:, :, 2, 2, 2, 2, 2, 2, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 1, 1, 1, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 1, 1, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 1, 1, 1, 1, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 1, 1, 1, 1, 2, 1] =
 0.0   0.0
 0.0  -0.00173367

[:, :, 1, 1, 2, 1, 1, 1, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 2, 1, 1, 1, 2, 1] =
 0.0  0.0
 0.0  0.00779143

[:, :, 1, 2, 2, 1, 1, 1, 2, 1] =
 0.0   0.0
 0.0  -0.0171258

[:, :, 2, 2, 2, 1, 1, 1, 2, 1] =
 0.0        -0.0166086
 0.0226151   0.0

[:, :, 1, 1, 1, 2, 1, 1, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 2, 1, 1, 2, 1] =
 0.0   0.0
 0.0  -0.0163391

[:, :, 1, 2, 1, 2, 1, 1, 2, 1] =
 0.0  0.0
 0.0  0.0471566

[:, :, 2, 2, 1, 2, 1, 1, 2, 1] =
  0.0        0.0527541
 -0.0688247  0.0

[:, :, 1, 1, 2, 2, 1, 1, 2, 1] =
 0.0   0.0
 0.0  -0.050527

[:, :, 2, 1, 2, 2, 1, 1, 2, 1] =
 0.0        -0.0805578
 0.0961739   0.0

[:, :, 1, 2, 2, 2, 1, 1, 2, 1] =
  0.0        0.0693627
 -0.0647359  0.0

[:, :, 2, 2, 2, 2, 1, 1, 2, 1] =
 -0.0288143  0.0
  0.0        0.0

[:, :, 1, 1, 1, 1, 2, 1, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 2, 1, 2, 1] =
 0.0  0.0
 0.0  0.0195825

[:, :, 1, 2, 1, 1, 2, 1, 2, 1] =
 0.0   0.0
 0.0  -0.062709

[:, :, 2, 2, 1, 1, 2, 1, 2, 1] =
 0.0        -0.0754775
 0.0961739   0.0

[:, :, 1, 1, 2, 1, 2, 1, 2, 1] =
 0.0  0.0
 0.0  0.0883824

[:, :, 2, 1, 2, 1, 2, 1, 2, 1] =
  0.0       0.151609
 -0.176777  0.0

[:, :, 1, 2, 2, 1, 2, 1, 2, 1] =
 0.0       -0.14484
 0.132026   0.0

[:, :, 2, 2, 2, 1, 2, 1, 2, 1] =
 0.0632262  0.0
 0.0        0.0

[:, :, 1, 1, 1, 2, 2, 1, 2, 1] =
 0.0   0.0
 0.0  -0.0583517

[:, :, 2, 1, 1, 2, 2, 1, 2, 1] =
 0.0       -0.115463
 0.128994   0.0

[:, :, 1, 2, 1, 2, 2, 1, 2, 1] =
  0.0       0.14484
 -0.126498  0.0

[:, :, 2, 2, 1, 2, 2, 1, 2, 1] =
 -0.06988  0.0
  0.0      0.0

[:, :, 1, 1, 2, 2, 2, 1, 2, 1] =
 0.0        -0.0693627
 0.0554347   0.0

[:, :, 2, 1, 2, 2, 2, 1, 2, 1] =
 0.0436437  0.0
 0.0        0.0

[:, :, 1, 2, 2, 2, 2, 1, 2, 1] =
 -0.0127685  0.0
  0.0        0.0

[:, :, 2, 2, 2, 2, 2, 1, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 1, 1, 2, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 1, 2, 2, 1] =
 0.0   0.0
 0.0  -0.012182

[:, :, 1, 2, 1, 1, 1, 2, 2, 1] =
 0.0  0.0
 0.0  0.0410656

[:, :, 2, 2, 1, 1, 1, 2, 2, 1] =
  0.0        0.0515139
 -0.0647359  0.0

[:, :, 1, 1, 2, 1, 1, 2, 2, 1] =
 0.0   0.0
 0.0  -0.0642187

[:, :, 2, 1, 2, 1, 1, 2, 2, 1] =
 0.0       -0.114809
 0.132026   0.0

[:, :, 1, 2, 2, 1, 1, 2, 2, 1] =
  0.0       0.115463
 -0.103799  0.0

[:, :, 2, 2, 2, 1, 1, 2, 2, 1] =
 -0.0518072  0.0
  0.0        0.0

[:, :, 1, 1, 1, 2, 1, 2, 2, 1] =
 0.0  0.0
 0.0  0.055671

[:, :, 2, 1, 1, 2, 1, 2, 2, 1] =
  0.0       0.114809
 -0.126498  0.0

[:, :, 1, 2, 1, 2, 1, 2, 2, 1] =
 0.0       -0.151609
 0.130587   0.0

[:, :, 2, 2, 1, 2, 1, 2, 2, 1] =
 0.0751842  0.0
 0.0        0.0

[:, :, 1, 1, 2, 2, 1, 2, 2, 1] =
  0.0        0.0805578
 -0.0634957  0.0

[:, :, 2, 1, 2, 2, 1, 2, 2, 1] =
 -0.0521005  0.0
  0.0        0.0

[:, :, 1, 2, 2, 2, 1, 2, 2, 1] =
 0.0160458  0.0
 0.0        0.0

[:, :, 2, 2, 2, 2, 1, 2, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 1, 2, 2, 2, 1] =
 0.0   0.0
 0.0  -0.0232168

[:, :, 2, 1, 1, 1, 2, 2, 2, 1] =
 0.0        -0.0515139
 0.0554347   0.0

[:, :, 1, 2, 1, 1, 2, 2, 2, 1] =
  0.0        0.0754775
 -0.0634957  0.0

[:, :, 2, 2, 1, 1, 2, 2, 2, 1] =
 -0.039332  0.0
  0.0       0.0

[:, :, 1, 1, 2, 1, 2, 2, 2, 1] =
 0.0        -0.0527541
 0.0406111   0.0

[:, :, 2, 1, 2, 1, 2, 2, 2, 1] =
 0.0358522  0.0
 0.0        0.0

[:, :, 1, 2, 2, 1, 2, 2, 2, 1] =
 -0.0122513  0.0
  0.0        0.0

[:, :, 2, 2, 2, 1, 2, 2, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 2, 2, 2, 2, 1] =
  0.0        0.0166086
 -0.0122503  0.0

[:, :, 2, 1, 1, 2, 2, 2, 2, 1] =
 -0.0124752  0.0
  0.0        0.0

[:, :, 1, 2, 1, 2, 2, 2, 2, 1] =
 0.00559752  0.0
 0.0         0.0

[:, :, 2, 2, 1, 2, 2, 2, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 2, 2, 2, 2, 2, 1] =
 -0.00121646  0.0
  0.0         0.0

[:, :, 2, 1, 2, 2, 2, 2, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 2, 2, 2, 2, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 2, 2, 2, 2, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 1, 1, 1, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 1, 1, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 1, 1, 1, 1, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 1, 1, 1, 1, 1, 2] =
 0.0  0.0
 0.0  0.00121646

[:, :, 1, 1, 2, 1, 1, 1, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 2, 1, 1, 1, 1, 2] =
 0.0   0.0
 0.0  -0.00559752

[:, :, 1, 2, 2, 1, 1, 1, 1, 2] =
 0.0  0.0
 0.0  0.0124752

[:, :, 2, 2, 2, 1, 1, 1, 1, 2] =
  0.0        0.0122503
 -0.0166086  0.0

[:, :, 1, 1, 1, 2, 1, 1, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 2, 1, 1, 1, 2] =
 0.0  0.0
 0.0  0.0122513

[:, :, 1, 2, 1, 2, 1, 1, 1, 2] =
 0.0   0.0
 0.0  -0.0358522

[:, :, 2, 2, 1, 2, 1, 1, 1, 2] =
 0.0        -0.0406111
 0.0527541   0.0

[:, :, 1, 1, 2, 2, 1, 1, 1, 2] =
 0.0  0.0
 0.0  0.039332

[:, :, 2, 1, 2, 2, 1, 1, 1, 2] =
  0.0        0.0634957
 -0.0754775  0.0

[:, :, 1, 2, 2, 2, 1, 1, 1, 2] =
 0.0        -0.0554347
 0.0515139   0.0

[:, :, 2, 2, 2, 2, 1, 1, 1, 2] =
 0.0232168  0.0
 0.0        0.0

[:, :, 1, 1, 1, 1, 2, 1, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 2, 1, 1, 2] =
 0.0   0.0
 0.0  -0.0160458

[:, :, 1, 2, 1, 1, 2, 1, 1, 2] =
 0.0  0.0
 0.0  0.0521005

[:, :, 2, 2, 1, 1, 2, 1, 1, 2] =
  0.0        0.0634957
 -0.0805578  0.0

[:, :, 1, 1, 2, 1, 2, 1, 1, 2] =
 0.0   0.0
 0.0  -0.0751842

[:, :, 2, 1, 2, 1, 2, 1, 1, 2] =
 0.0       -0.130587
 0.151609   0.0

[:, :, 1, 2, 2, 1, 2, 1, 1, 2] =
  0.0       0.126498
 -0.114809  0.0

[:, :, 2, 2, 2, 1, 2, 1, 1, 2] =
 -0.055671  0.0
  0.0       0.0

[:, :, 1, 1, 1, 2, 2, 1, 1, 2] =
 0.0  0.0
 0.0  0.0518072

[:, :, 2, 1, 1, 2, 2, 1, 1, 2] =
  0.0       0.103799
 -0.115463  0.0

[:, :, 1, 2, 1, 2, 2, 1, 1, 2] =
 0.0       -0.132026
 0.114809   0.0

[:, :, 2, 2, 1, 2, 2, 1, 1, 2] =
 0.0642187  0.0
 0.0        0.0

[:, :, 1, 1, 2, 2, 2, 1, 1, 2] =
  0.0        0.0647359
 -0.0515139  0.0

[:, :, 2, 1, 2, 2, 2, 1, 1, 2] =
 -0.0410656  0.0
  0.0        0.0

[:, :, 1, 2, 2, 2, 2, 1, 1, 2] =
 0.012182  0.0
 0.0       0.0

[:, :, 2, 2, 2, 2, 2, 1, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 1, 1, 2, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 1, 2, 1, 2] =
 0.0  0.0
 0.0  0.0127685

[:, :, 1, 2, 1, 1, 1, 2, 1, 2] =
 0.0   0.0
 0.0  -0.0436437

[:, :, 2, 2, 1, 1, 1, 2, 1, 2] =
 0.0        -0.0554347
 0.0693627   0.0

[:, :, 1, 1, 2, 1, 1, 2, 1, 2] =
 0.0  0.0
 0.0  0.06988

[:, :, 2, 1, 2, 1, 1, 2, 1, 2] =
  0.0      0.126498
 -0.14484  0.0

[:, :, 1, 2, 2, 1, 1, 2, 1, 2] =
 0.0       -0.128994
 0.115463   0.0

[:, :, 2, 2, 2, 1, 1, 2, 1, 2] =
 0.0583517  0.0
 0.0        0.0

[:, :, 1, 1, 1, 2, 1, 2, 1, 2] =
 0.0   0.0
 0.0  -0.0632262

[:, :, 2, 1, 1, 2, 1, 2, 1, 2] =
 0.0      -0.132026
 0.14484   0.0

[:, :, 1, 2, 1, 2, 1, 2, 1, 2] =
  0.0       0.176777
 -0.151609  0.0

[:, :, 2, 2, 1, 2, 1, 2, 1, 2] =
 -0.0883824  0.0
  0.0        0.0

[:, :, 1, 1, 2, 2, 1, 2, 1, 2] =
 0.0        -0.0961739
 0.0754775   0.0

[:, :, 2, 1, 2, 2, 1, 2, 1, 2] =
 0.062709  0.0
 0.0       0.0

[:, :, 1, 2, 2, 2, 1, 2, 1, 2] =
 -0.0195825  0.0
  0.0        0.0

[:, :, 2, 2, 2, 2, 1, 2, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 1, 2, 2, 1, 2] =
 0.0  0.0
 0.0  0.0288143

[:, :, 2, 1, 1, 1, 2, 2, 1, 2] =
  0.0        0.0647359
 -0.0693627  0.0

[:, :, 1, 2, 1, 1, 2, 2, 1, 2] =
 0.0        -0.0961739
 0.0805578   0.0

[:, :, 2, 2, 1, 1, 2, 2, 1, 2] =
 0.050527  0.0
 0.0       0.0

[:, :, 1, 1, 2, 1, 2, 2, 1, 2] =
  0.0        0.0688247
 -0.0527541  0.0

[:, :, 2, 1, 2, 1, 2, 2, 1, 2] =
 -0.0471566  0.0
  0.0        0.0

[:, :, 1, 2, 2, 1, 2, 2, 1, 2] =
 0.0163391  0.0
 0.0        0.0

[:, :, 2, 2, 2, 1, 2, 2, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 2, 2, 2, 1, 2] =
 0.0        -0.0226151
 0.0166086   0.0

[:, :, 2, 1, 1, 2, 2, 2, 1, 2] =
 0.0171258  0.0
 0.0        0.0

[:, :, 1, 2, 1, 2, 2, 2, 1, 2] =
 -0.00779143  0.0
  0.0         0.0

[:, :, 2, 2, 1, 2, 2, 2, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 2, 2, 2, 2, 1, 2] =
 0.00173367  0.0
 0.0         0.0

[:, :, 2, 1, 2, 2, 2, 2, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 2, 2, 2, 2, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 2, 2, 2, 2, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 1, 1, 1, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 1, 1, 1, 1, 2, 2] =
 0.0   0.0
 0.0  -0.00508031

[:, :, 1, 2, 1, 1, 1, 1, 2, 2] =
 0.0  0.0
 0.0  0.0178488

[:, :, 2, 2, 1, 1, 1, 1, 2, 2] =
  0.0        0.0232168
 -0.0288143  0.0

[:, :, 1, 1, 2, 1, 1, 1, 2, 2] =
 0.0   0.0
 0.0  -0.0300308

[:, :, 2, 1, 2, 1, 1, 1, 2, 2] =
 0.0        -0.055671
 0.0632262   0.0

[:, :, 1, 2, 2, 1, 1, 1, 2, 2] =
  0.0        0.0583517
 -0.0518072  0.0

[:, :, 2, 2, 2, 1, 1, 1, 2, 2] =
 -0.0268122  0.0
  0.0        0.0

[:, :, 1, 1, 1, 2, 1, 1, 2, 2] =
 0.0  0.0
 0.0  0.0300308

[:, :, 2, 1, 1, 2, 1, 1, 2, 2] =
  0.0      0.0642187
 -0.06988  0.0

[:, :, 1, 2, 1, 2, 1, 1, 2, 2] =
 0.0        -0.0883824
 0.0751842   0.0

[:, :, 2, 2, 1, 2, 1, 1, 2, 2] =
 0.0448849  0.0
 0.0        0.0

[:, :, 1, 1, 2, 2, 1, 1, 2, 2] =
  0.0       0.050527
 -0.039332  0.0

[:, :, 2, 1, 2, 2, 1, 1, 2, 2] =
 -0.0334649  0.0
  0.0        0.0

[:, :, 1, 2, 2, 2, 1, 1, 2, 2] =
 0.0107416  0.0
 0.0        0.0

[:, :, 2, 2, 2, 2, 1, 1, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 1, 2, 1, 2, 2] =
 0.0   0.0
 0.0  -0.0178488

[:, :, 2, 1, 1, 1, 2, 1, 2, 2] =
 0.0        -0.0410656
 0.0436437   0.0

[:, :, 1, 2, 1, 1, 2, 1, 2, 2] =
  0.0        0.062709
 -0.0521005  0.0

[:, :, 2, 2, 1, 1, 2, 1, 2, 2] =
 -0.0334649  0.0
  0.0        0.0

[:, :, 1, 1, 2, 1, 2, 1, 2, 2] =
 0.0        -0.0471566
 0.0358522   0.0

[:, :, 2, 1, 2, 1, 2, 1, 2, 2] =
 0.0328197  0.0
 0.0        0.0

[:, :, 1, 2, 2, 1, 2, 1, 2, 2] =
 -0.0116885  0.0
  0.0        0.0

[:, :, 2, 2, 2, 1, 2, 1, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 2, 2, 1, 2, 2] =
  0.0        0.0171258
 -0.0124752  0.0

[:, :, 2, 1, 1, 2, 2, 1, 2, 2] =
 -0.0131734  0.0
  0.0        0.0

[:, :, 1, 2, 1, 2, 2, 1, 2, 2] =
 0.00616033  0.0
 0.0         0.0

[:, :, 2, 2, 1, 2, 2, 1, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 2, 2, 2, 1, 2, 2] =
 -0.00144039  0.0
  0.0         0.0

[:, :, 2, 1, 2, 2, 2, 1, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 2, 2, 2, 1, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 2, 2, 2, 1, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 1, 1, 2, 2, 2] =
 0.0  0.0
 0.0  0.00508031

[:, :, 2, 1, 1, 1, 1, 2, 2, 2] =
  0.0        0.012182
 -0.0127685  0.0

[:, :, 1, 2, 1, 1, 1, 2, 2, 2] =
 0.0        -0.0195825
 0.0160458   0.0

[:, :, 2, 2, 1, 1, 1, 2, 2, 2] =
 0.0107416  0.0
 0.0        0.0

[:, :, 1, 1, 2, 1, 1, 2, 2, 2] =
  0.0        0.0163391
 -0.0122513  0.0

[:, :, 2, 1, 2, 1, 1, 2, 2, 2] =
 -0.0116885  0.0
  0.0        0.0

[:, :, 1, 2, 2, 1, 1, 2, 2, 2] =
 0.00438211  0.0
 0.0         0.0

[:, :, 2, 2, 2, 1, 1, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 2, 1, 2, 2, 2] =
 0.0         -0.00779143
 0.00559752   0.0

[:, :, 2, 1, 1, 2, 1, 2, 2, 2] =
 0.00616033  0.0
 0.0         0.0

[:, :, 1, 2, 1, 2, 1, 2, 2, 2] =
 -0.00303256  0.0
  0.0         0.0

[:, :, 2, 2, 1, 2, 1, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 2, 2, 1, 2, 2, 2] =
 0.00078674  0.0
 0.0         0.0

[:, :, 2, 1, 2, 2, 1, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 2, 2, 1, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 2, 2, 1, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 1, 2, 2, 2, 2] =
  0.0         0.00173367
 -0.00121646  0.0

[:, :, 2, 1, 1, 1, 2, 2, 2, 2] =
 -0.00144039  0.0
  0.0         0.0

[:, :, 1, 2, 1, 1, 2, 2, 2, 2] =
 0.00078674  0.0
 0.0         0.0

[:, :, 2, 2, 1, 1, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 2, 1, 2, 2, 2, 2] =
 -0.000268478  0.0
  0.0          0.0

[:, :, 2, 1, 2, 1, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 2, 1, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 2, 1, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 1, 2, 2, 2, 2, 2] =
 4.45483e-5  0.0
 0.0         0.0

[:, :, 2, 1, 1, 2, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 1, 2, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 1, 2, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 1, 2, 2, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1, 2, 2, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2, 2, 2, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2, 2, 2, 2, 2, 2, 2] =
 0.0  0.0
 0.0  0.0
````

## Part (a) — bipartition after the first site (1 | 9)

````julia
bp_a  = bipartition(Partition(sites[1]), A)
res_a = tensor_svd(A, bp_a, KeepAbove(1e-6); normalize=true)
(bipartition = "1 | 9", Schmidt_rank = size(res_a.Σ.data, 1))
````

````
(bipartition = "1 | 9", Schmidt_rank = 2)
````

## Part (b) — bipartition at the middle (5 | 5)

````julia
bp_b  = bipartition(Partition(sites[1:N÷2]...), A)
res_b = tensor_svd(A, bp_b, KeepAbove(1e-6); normalize=true)
(bipartition = "5 | 5", Schmidt_rank = size(res_b.Σ.data, 1))
````

````
(bipartition = "5 | 5", Schmidt_rank = 24)
````

## Entanglement entropy profile

The **von Neumann entanglement entropy** of a bipartition quantifies how much
the two halves are correlated.  Given Schmidt coefficients
``\lambda_i`` (normalised singular values with ``\sum_i \lambda_i^2 = 1``),
the entropy is:

```math
S = -\sum_i \lambda_i^2 \log_b \lambda_i^2,
```

with base ``b = 2`` for **bits** or ``b = e`` for **nats**.
A product state has ``S = 0``; a maximally entangled state across a
2^k|2^k bipartition has ``S = k`` bits.

**Your implementation:** fill in the formula below.
- Choose a logarithm base (and note what units the result is in).
- Guard against ``\lambda_i = 0`` to avoid ``0 \log 0``.

````julia
function entanglement_entropy(λ::AbstractVector{<:Real})
    p = λ .^ 2
    return -sum(pᵢ * log2(pᵢ) for pᵢ in p if pᵢ > 0)
end
````

````
entanglement_entropy (generic function with 1 method)
````

Once `entanglement_entropy` is implemented, the cell below sweeps every
bipartition boundary and plots the resulting entropy profile.

````julia
entropies = map(1:N-1) do i
    bp  = bipartition(Partition(sites[1:i]...), A)
    res = tensor_svd(A, bp, KeepMachineEps(); normalize=true)
    entanglement_entropy(diag(res.Σ.data))
end

fig = Figure(size=(620, 360))
ax  = Axis(fig[1, 1];
    title  = "Entanglement entropy across bipartitions",
    xlabel = "boundary position  i  (site i | i+1 … N)",
    ylabel = "S (bits)",
    xticks = 1:N-1,
)
lines!(ax, 1:N-1, entropies; color=:teal, linewidth=2.5)
scatter!(ax, 1:N-1, entropies; color=:teal, markersize=9)
fig
````

```@raw html
<img width=620 height=360 style='object-fit: contain; height: auto;' src="data:image/png;base64, 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z950E1V8fRERERERkYtX1c+P4TfeyMtr1nAhvl5evNimDSLyvwxERK5hsdWrM7NrVx6cP5/84mLOp4fVyvO33IKIiIiIiFy6f996K78cPcqc1FTOx88wmNm1K9HVqiEi/8tAROQad39MDA2Cg/n3t9+yZPduCktKKBVbvTpP33gjD8XFISIiIiIiZeNlMvHZfffxj4gIxq1bx47ffqOUt9nMnVFRvNy2Lddfdx0i8n8ZiIi4gUY1arAwIYHfCwtZv2sXp/PyuCE8nPpBQYiIiIiIyF9nMpnoGx9P3/h4snJy2JKZSWV/f1pFRVHR2xsROT8DERE3UtHbG2twMAUFBdSsVAkREREREbn8aleqhPm66/Dx8aGitzci8ucMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMREREREREREREpMwMRFyYTCZEREREREREROTiGIiIiIiIiIiIiEiZGYj8f3a7HXd25MgRatasSY0aNTh8+DDiueLj49m6dSspKSnExcUhnmn48OGMHz+ecePGMWzYMMQzrVixgttvv52OHTuyfPlyxHOZTCYc7HY74rluv/12VqxYwfLly+nYsSPimcaPH8/w4cMZNmwY48aNQzzT1q1biY+PJy4ujpSUFMRz1axZkyNHjnD48GFq1KiBlI2BiIiIiIiIiIiIlJmBiIiIiIiIiIiIlJmBiIiIiIiIiIiIlJmBiIiIiIiIiIiIlJmBiIiIiIiIiIiIlJmBiIiIiIiIiIiIlJmBiIiIiIiIiIiIlJmBiIiIiIiIiIiIlJmBiIiIiIiIiIiIlJmBiIeoUaMGdrsd8XwpKSmI5xs3bhzjxo1DPFvHjh2x2+2I57Pb7YjnW758OeL5hg0bxrBhwxDPFhcXh91uRzzf4cOHkb/OQERERERERERERMrMQERERERERERERMrMQERERERERERERMrMQERERERERERERMrMQERERERERERERMrMQERERERERERERMrMQERERERERERE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L7/8EleLFi1i0qRJfPbZZzRu3BhXycnJ9OjRg8OHD+NszZo1TJ8+naSkJJKTkwkODqZUfn4+iYmJOERERFCvXj02btxIYmIipQoKCkhMTMThk08+ISoqivz8fBITE3GIiIigXr16ONu6dSsJCQmkpaXhbN26dcyaNYuPP/6Yjz76iOuuuw5nNpuNxMRELBYLnTp1okuXLnz//fc4W7p0KRMnTmTevHnccccdXC75+fkkJibiEBERQb169bhcfv31V5YuXYqfnx933HEH55KcnEyPHj04fPgwztasWcP06dNJSkoiOTmZ4OBgSuXn55OYmIhDREQE9erVY+PGjSQmJlKqoKCAxMREHD755BOioqIolZqaSvfu3dmxYwfOVq9ezXvvvcftt9/O3LlzCQgI4FKMHTuWn376iWeeeYZWrVpxNSxYsIDXX3+dFi1a0K1bNy6kqKgIk8mEl5cXF2KxWNizZw/OEhIS2LBhA56sZs2ajB8/nkcffZSffvqJt956i6FDh3KpwsPDiY+PJyUlhTVr1tCuXTtEROTyMhARESkHunfvTmZmJrfccgsdO3YkICCAdevW8cUXX3D69Gl69OhBcXExvXr1wtm0adPo168fDtdddx3du3cnPj6erKwsvvrqK1JSUhg7dixZWVnMmTOHK+G1117j3Xff5Y477uD+++8nNDSUn376iTFjxnDs2DFeeeUVOnXqRKtWrSi1efNmbr31VgoLCzGbzXTt2pUbb7yRM2fOsGzZMn744QfuueceioqKOJdvv/2WIUOGUFxcTEhICA8//DARERHk5+eTkpLCJ598wunTp+nVqxfp6enUqFGDcxkwYAAbNmzA39+fyMhIbrrpJiIjI4mPjyclJYUvvviCHj16cC6zZs3CoXPnzlStWpVLMWfOHJKTk/Hz86NXr160bNmSffv28fXXX7N9+3Y+//xz9u/fz7p16zCZTJQqKCigZcuWpKen49CxY0fatWtHlSpV2LhxI59++im7du2iVatWbNy4kbi4OErt37+fbt26cezYMYKCgnjkkUewWq3k5eWxYsUKFixYgM1m47HHHmP+/Pn8mfj4eEaPHs2WLVuYP38+Pj4+jBw5Eof4+HguZMeOHbRo0YL8/Hy8vb1JSEigZcuW/P777yQnJ7NkyRKWLl1K8+bNSUtLw8/PD1d2u51evXqxbt06Bg4c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J/v27ePixYs4Ozvj7+/PwIED8fHxoT5WrlzJtWvX8PT05Ofy2muvMXjwYJo1a8ZPMZvNzJs3j/uxePFiSkpKcHZ25n6NHz+eAQMGYGjbti01adeuHWlpaRicnZ2pzcaNG7l9+zYGW1tb7tX06dMZPXo0TZo04W74+PiQkZFBUVERhsDAQKo7c+YMR44cwd3dnd69eyMiIg3PjIiIyEPKyckJi8VCJbPZzAsvvMALL7zAvWrSpAmjRo1i1KhR3CsvLy/efPNNatKoUSMsFgvV2draYrFY+ClBQUHUxc7OjqFDhzJ06FBqEhwcTG2cnJyYMGECdQkICCAgIICqnJ2dsVgs3K3S0lI2b96MISoqChsbGxqKm5sbCQkJ1EdISAghISHcKycnJ15++WXulq2tLRaLhdo0btyYYcOGUZ2trS0Wi4W6mEwm+vXrR79+/bhbzs7OWCwWfkqPHj1oaLa2tlgsFh6U119/nZSUFBITE4mIiKAmTk5OvPzyy9wtW1tbLBYLtWncuDHDhg2jLmazmYiICCIiImgooaGh/Nx8fX3x9fXl59KhQwcaio+PDz4+PtSlWbNmWCwWfkrXrl25H23btqVt27bci/bt21OXtWvXYrVamTBhAmazGRERaXhmRERERP4Xbdq0iaKiIgwxMTGIPCgDBw7E39+fbdu2UVJSgpOTEyKPgsTERBwcHBg/fjwiIvJgmBERERH5md28eZPGjRvz3XffMW3aNAzdu3enXbt2iDwoJpOJmTNnEhMTw4oVK5g+fToiD7uUlBRycnKYMmUKzs7OiIjIg2FGRERE5Gc2f/585s6dS1lZGQaTycScOXMQedCio6NJTExk/vz5vPrqqzg5OSHyMJsxYwbe3t7MmjULERF5cMyIiIiI/C8oKyvD8Nhjj7FkyRIsFgsiP4cVK1YQFBTEe++9x9y5cxF5WG3evJnDhw+TnJzMY489hoiIPDhmRERERH5mb731Fn379qWkpISnn34aFxcXGsLUqVOJiorCz88Pkdr4+Phw4sQJSktLkf+padOmHDt2DAcHB+S/X+fOncnKyiIwMBAREXmwzIiIiIj8zBo1akT37t1paL1790bkbvj4+CD/P1tbWzp06IA8HDw9PRERkZ+HGREREREREREREak3MyIiIiIiIiIiIlJvZkRERERERERERKTezIiIiIiIiIiIiEi9mREREREREREREZF6MyMiIiIiIiIiIiL1ZkZERERERERERETqzYyIiIiIiIiIiIjUmxkRERERERERERGpNzMiIiIiIiIiIiJSb2ZERERERERERESk3syIiIiIiIiIiIhIvZkRERERERERERGRejMjIiIiIiIiIiIi9WZGRERERERERERE6s2MiIiIiIiIiIiI1JsZERERERERERERqTczIiIiIiIiIiIiUm9mREREREREREREpN7MiIiIiIiIiIiISL2ZERERERERERERkXozIyIiIiIiIiIiIvVmRkREREREREREROrNjIiIiIiIiIiIiNSbGREREREREREREak3MyIiIiIiIiIiIlJvZkRERERERERERKTezIiIiIiIiIiIiEi9mREREREREREREZF6+3+evpXy9KqoVAAAAABJRU5ErkJggg=="/>
```

## Notes

- A small Schmidt rank means the bipartition carries *low entanglement*: the
  state can be represented faithfully with few terms in its Schmidt expansion.
- The entropy at the middle bipartition is typically the highest because the
  two halves can exchange information across every pair of consecutive bonds
  between them.
- Both `normalize=true` and `KeepMachineEps()` matter here: the former
  produces proper Schmidt coefficients ``\sum_i \lambda_i^2 = 1``, while the
  latter discards floating-point noise that would otherwise inflate the
  entropy by tiny non-zero terms.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

