
# # Task 1.4 — SVD an image

# !!! question "Task 1.4"
#     Reproduce an SVD-based image compression.  Perform an SVD on each colour
#     channel separately.

DATA_ROOT = normpath(joinpath(@__FILE__, "..", "..", "data"))
FPATH_IMG = joinpath(DATA_ROOT, "Bahkauv.png")
#--

using LinearAlgebra, CairoMakie, FileIO, PNGFiles, ColorTypes, FixedPointNumbers

img = load(FPATH_IMG)    # Matrix{RGB{N0f8}}

# Extract the three colour channels as Float64 matrices.

R = Float64.(red.(img))
G = Float64.(green.(img))
B = Float64.(blue.(img))

# ## Rank-r compression

# A rank-``r`` approximation to a channel matrix ``M`` is:
#
# ```math
# M_r = \sum_{i=1}^{r} \sigma_i u_i v_i^\top,
# ```
#
# retaining only the ``r`` largest singular values.  The compression ratio
# (number of stored floats) compared to the original is approximately
# ``r(m + n + 1) / (mn)``.

function compress_channel(M::AbstractMatrix{<:Real}, r::Int)
    F  = svd(M)
    Mr = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
    return clamp.(Mr, 0.0, 1.0)
end

function compress_image(R, G, B, r::Int)
    RGB.(compress_channel(R, r), compress_channel(G, r), compress_channel(B, r))
end
#--

# ## Comparison at increasing ranks

ranks = [5, 20, 50, minimum(size(R))]

fig  = Figure(size=(220 * (1 + length(ranks)), 280))
ax0  = Axis(fig[1, 1]; aspect=DataAspect(), title="original")
hidedecorations!(ax0)
image!(ax0, rotr90(img))

for (i, r) in enumerate(ranks)
    m, n    = size(R)
    ratio   = round(r * (m + n + 1) / (m * n) * 100; digits=1)
    ax      = Axis(fig[1, i+1]; aspect=DataAspect(), title="rank $r  ($ratio% of pixels)")
    hidedecorations!(ax)
    image!(ax, rotr90(compress_image(R, G, B, r)))
end
fig

# ## Notes

# - Each channel is an independent ``m \times n`` matrix — there is no coupling
#   between R, G, and B in a standard SVD compression.  The singular vectors
#   of R may not align with those of G.
# - The singular values of a typical photographic channel decay rapidly: a small
#   number of large ``\sigma_i`` capture most of the "energy"
#   (``\|M\|_F^2 = \sum_i \sigma_i^2``), and the remaining ones correspond to
#   fine texture and noise.
# - The compression ratio ``r(m+n+1)/(mn)`` crosses 1 at roughly
#   ``r \approx mn/(m+n)``, so SVD compression is not worthwhile for
#   high-rank images unless the singular value spectrum decays steeply.
