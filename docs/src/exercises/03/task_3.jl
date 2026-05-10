# # Task 3.3 — Checking the Normalization

# !!! question "Task 3.3 — Checking the Normalization"
#     Write a function that receives a MPS and checks at each site for left (A) and right (B) normalization. Think about a good measure telling you how far away you are from unity in these normalizations.

using Qritical: QriticalUtils, factorize_with_svd
using LinearAlgebra
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI = normpath(joinpath(DATA_ROOT, "psi.jls"))

tolerance = 1e-6
#--

# !!! subquestion
#     **A)** Left normalization: for each site tensor ``A^{[i]}``, check that ``\sum_\sigma {A^{[i]}}^\dagger_\sigma A^{[i]}_\sigma = \mathbb{I}``

# !!! subquestion
#     **B)** Right normalization: for each site tensor ``B^{[i]}``, check that ``\sum_\sigma B^{[i]}_\sigma {B^{[i]}}^\dagger_\sigma = \mathbb{I}``
