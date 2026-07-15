# Using the Glossary System

The Qritical.jl documentation now includes an interactive glossary with hover tooltips and clickable links to detailed definitions.

## Overview

The glossary system provides:
- ✅ **Hover tooltips** — Brief definitions appear on hover
- ✅ **Clickable links** — Click to jump to full definition in Glossary page
- ✅ **Semantic HTML** — Uses `<abbr>` tags for accessibility
- ✅ **Light/dark mode support** — Tooltips adapt to theme

## How to Use

### Method 1: Simple Markdown Links (Recommended)

In any markdown page, simply wrap a term in the `@ref` cross-reference syntax pointing to the Glossary:

```markdown
An [`MPS`](@ref Glossary#mps) is a compressed representation of quantum states.
The [`Schmidt Rank`](@ref Glossary#schmidt-rank) measures entanglement.
```

This automatically:
- Creates a clickable link to the glossary definition
- Handles anchor generation (spaces → hyphens, lowercase)
- Works in both source .md and Literate.jl files

### Method 2: Using the Helper Function (Advanced)

If you're generating documentation programmatically, use the `glossary_helper.jl` utilities:

```julia
# In docs/make.jl or custom scripts:
include("glossary_helper.jl")

# Link with auto-filled brief definition
glossary_term_with_brief("MPS")  
# → [MPS](@ref Glossary#mps) with title="Matrix Product State..."

# Or specify custom brief definition
glossary_term("Bond Dimension"; brief="Size of virtual bonds controlling MPS accuracy")
```

### Method 3: Raw HTML (Special Cases)

For fine-grained control, use the `glossary_html` function:

```julia
glossary_html("MPS", "Matrix Product State")
# → <abbr title="Matrix Product State"><a href="#mps">MPS</a></abbr>
```

## Customizing the Glossary

### Adding New Terms

1. **Edit** `docs/src/references/glossary.md`
2. **Add your term** under the appropriate letter section:
   ```markdown
   ### Your New Term
   
   Definition goes here. Use LaTeX: ``\sigma_i`` for inline math,
   and ```math blocks for display equations.
   ```
3. **Add brief definition** to `GLOSSARY_BRIEFS` dict in `glossary_helper.jl`:
   ```julia
   "Your New Term" => "Brief definition for tooltip",
   ```
4. **Use in docs** via `@ref` links (auto-discovery of brief definition)

### Styling

CSS is in `docs/assets/custom.css`:
- `abbr` — Base styling (dotted underline, help cursor)
- `abbr[title]:hover::after` — Tooltip appearance
- Light/dark mode variants included

Customize colors, spacing, or behavior there.

## Examples in Documentation

### Physics Motivation Sections

```markdown
# SVD & Truncation

The [`Schmidt decomposition`](@ref Glossary#schmidt-decomposition) is the 
fundamental tool of tensor-network methods. For a bipartite quantum state, 
the [`Schmidt Rank`](@ref Glossary#schmidt-rank) determines entanglement, 
and **truncation** keeps only the top ``r`` values.
```

### API Docstrings

```julia
"""
    to_mps(ψ::AbstractTensor; kwargs...) -> FiniteMPS

Decompose a full state tensor into an [`MPS`](@ref Glossary#mps) via 
iterated SVD with controlled [`truncation`](@ref Glossary#truncation).

See also: [`do_svd`](@ref), [`Glossary`](@ref).
"""
function to_mps(ψ, kwargs...) ... end
```

### Exercise Notebooks

```julia
#md # # Exercise: Building an MPS
#md # 
#md # In this exercise we'll construct a [`Matrix Product State`](@ref Glossary#matrix-product-state)
#md # from a full state tensor using SVD. The [`Bond Dimension`](@ref Glossary#bond-dimension)
#md # controls the accuracy-to-cost tradeoff.
```

## Accessibility

The glossary system follows best practices:
- Uses semantic `<abbr>` HTML tag
- Keyboard accessible (tab to links, click to navigate)
- Screen reader friendly (title attributes read aloud)
- Works without JavaScript (links still function)
- WCAG 2.1 AA compliant contrast

## Anchor Format

Glossary anchors are generated automatically:
- Spaces → hyphens: `"Schmidt Rank"` → `#schmidt-rank`
- Lowercase: `"MPS"` → `#mps`
- Special chars removed

When writing `@ref` links, use this format:
```markdown
[`Term Name`](@ref Glossary#term-name)
```

## Tips

1. **Keep briefs short** — Under 80 chars for clean tooltips
2. **Link early, link often** — First mention of technical terms especially
3. **Use in introductions** — Index/API pages benefit most from glossary context
4. **Avoid over-linking** — Every occurrence is overkill; 1-2 per page is good
5. **Update when terminology changes** — Keep glossary synchronized with code

## Testing

To verify glossary works:
```bash
cd docs
julia --project=. -e 'include("make.jl")'
# Open build/index.html and navigate to References > Glossary
# Try hovering over any abbreviation (appears as dotted underline)
# Try clicking to navigate
```

---

**Created:** 2026-07-04  
**Last updated:** 2026-07-04
