# Helper utilities for linking to glossary terms in markdown

"""
    glossary_term(term::String; brief::String="") -> String

Create a markdown link to a glossary term with optional hover tooltip.

# Arguments

  - `term`: The glossary term to link (e.g., "MPS", "Entanglement Entropy")
  - `brief`: Brief definition to show on hover (optional)

# Example

```julia
glossary_term(\"MPS\"; brief=\"Matrix Product State\")
glossary_term(\"Schmidt Rank\")
```

# Returns

Markdown string with link to glossary and HTML title attribute for hover tooltip.
"""
function glossary_term(term::String; brief::String="")
    anchor = replace(lowercase(term), " " => "-")
    title = if isempty(brief)
        ""
    else
        " title=\"$brief\""
    end
    return "[$term](@ref Glossary#$(anchor))$title"
end

"""
    glossary_html(term::String, brief::String="") -> String

Generate HTML for a glossary term with hover tooltip and link.

Useful when you need direct HTML control in markdown documents.

# Example

```html
\$(glossary_html("MPS", "Matrix Product State"))
```
"""
function glossary_html(term::String, brief::String="")
    anchor = replace(lowercase(term), " " => "-")
    title = if isempty(brief)
        ""
    else
        " title=\"$brief\""
    end
    return "<abbr$(title)><a href=\"#$(anchor)\">$term</a></abbr>"
end

"""
    @gref term

Macro for quickly referencing a glossary term.
Translates to a link with hover tooltip if a brief definition is known.

# Example

```julia
@gref \"MPS\"
@gref \"Entanglement Entropy\"
```
"""
macro gref(term)
    return esc(glossary_term(term))
end

# Map of common glossary terms to brief definitions for hover tooltips
const GLOSSARY_BRIEFS = Dict(
    "Area Law" => "Entanglement entropy scales with boundary, not volume",
    "MPS" => "Matrix Product State: compressed tensor representation of quantum state",
    "MPO" => "Matrix Product Operator: compressed operator representation",
    "SVD" => "Singular Value Decomposition: matrix factorization tool",
    "Schmidt Rank" => "Number of nonzero singular values in Schmidt decomposition",
    "Schmidt Values" => "Singular values in Schmidt decomposition of state",
    "Schmidt Decomposition" => "Factorization of bipartite state into left and right parts",
    "Bond Dimension" => "Size of virtual bonds in MPS (controls accuracy and cost)",
    "Entanglement Entropy" => "Measure of quantum entanglement across a bipartition",
    "Entanglement Spectrum" => "Eigenvalues of reduced density matrix",
    "Canonical Form" => "MPS gauge where tensors are orthonormal (left or right isometric)",
    "Orthogonality Centre" => "Tensor in MPS where orthogonality property is concentrated",
    "Truncation" => "Discarding small singular values to reduce bond dimension",
    "TEBD" => "Time-Evolving Block Decimation: algorithm for time evolution",
    "Spectrum" => "Set of singular or eigenvalues from SVD or diagonalization",
    "Gauge" => "Redundant degree of freedom in MPS representation",
    "Index Variance" => "Direction (Upper/Lower) of tensor leg in von Delft convention",
    "DoF" => "Degree of Freedom: elementary quantum system on single site",
    "CCR" => "Canonical Commutation Relation (bosons, spins)",
    "CAR" => "Canonical Anti-Commutation Relation (fermions)",
    "Block-Sparse" => "Tensor stored as independent blocks by symmetry sector",
    "MBL" => "Many-Body Localization: localized phase at strong disorder",
)

"""
    glossary_term_with_brief(term::String) -> String

Create a glossary link with automatically-filled brief definition from GLOSSARY_BRIEFS.

# Example

```julia
glossary_term_with_brief(\"MPS\")  # Link with \"Matrix Product State...\" as hover text
glossary_term_with_brief(\"TEBD\")  # Link with \"Time-Evolving Block Decimation...\" tooltip
```
"""
function glossary_term_with_brief(term::String)::String
    brief = get(GLOSSARY_BRIEFS, term, "")
    return glossary_term(term; brief=brief)
end

# Export the functions for use in markdown
export glossary_term, glossary_html, glossary_term_with_brief
