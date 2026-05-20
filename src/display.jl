# ── LaTeX index notation ──────────────────────────────────────────────────────

# Internal: produce the LaTeX fragment for a single index leg.
# is_first controls whether to prepend {} for positional spacing.
#
# Examples (positional notation):
#   (σ UpIndex,   is_first=true)  → "^{\\sigma}"
#   (α DownIndex, is_first=false) → "{}\_{\\alpha}"
#   (β UpIndex,   is_first=false) → "{}^{\\beta}"
function _index_latex_fragment(idx::AbstractIndex, is_first::Bool)
    # TODO: implement
    # 1. Convert idx.label (a Symbol) to a string — use string(idx.label)
    # 2. Build the prefix: "" when is_first, "{}" otherwise
    # 3. If idx.dir == UpIndex  → prefix * "^{sym}"
    #    If idx.dir == DownIndex → prefix * "_{sym}"
end

"""
    index_latex(t::IndexedTensor, name::AbstractString) → LaTeXString

Return a `LaTeXString` of `t` in positional index notation. Each leg appears
in leg order: `UpIndex` legs as superscripts (`^{}`), `DownIndex` legs as
subscripts (`_{}`). Consecutive indices of the same kind are separated by
`{}` to preserve positional meaning.

Compatible with Makie axis labels and annotations.

# Examples
```jldoctest
julia> using HalfIntegers: half
julia> σ = PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex);
julia> α = BondIndex(:α, 1, 2, 4, DownIndex);
julia> β = BondIndex(:β, 2, 3, 3, DownIndex);
julia> A = IndexedTensor(ones(2, 4, 3), (σ, α, β));
julia> index_latex(A, "A")
L"\$A^{\\sigma}{}_{\\alpha}{}_{\\beta}\$"
```
"""
function index_latex(t::IndexedTensor, name::AbstractString)
    parts = [_index_latex_fragment(idx, i == 1) for (i, idx) in enumerate(t.indices)]
    return LaTeXString("\$" * name * join(parts) * "\$")
end

# ── Jupyter / Pluto rich display ──────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/latex", t::IndexedTensor)
    write(io, "\$\$" * index_latex(t, "\\mathcal{T}").s * "\$\$")
end
