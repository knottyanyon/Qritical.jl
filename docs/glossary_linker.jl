# Custom Documenter hook to convert glossary references to proper links
# Syntax: {glossary:term name} converts to [term name](../references/glossary.md#anchor)

"""
    glossary_link_preprocessor(content::String) -> String

Convert glossary reference syntax {glossary:term name} to proper markdown links.
Supports both simple terms and multi-word terms.

# Syntax

Use `{glossary:term name}` in markdown to create a glossary link:
- `{glossary:MPS}` → [`MPS`](../references/glossary.md#mps)
- `{glossary:Bond Dimension}` → [`Bond Dimension`](../references/glossary.md#bond-dimension)

The anchor is auto-generated from term name:
- Lowercase
- Spaces → hyphens
- Special chars removed

# Example

In markdown:
```
An {glossary:MPS} is a compressed state representation.
The {glossary:Schmidt Rank} measures entanglement.
```
"""
function glossary_link_preprocessor(content::String)::String
    # Pattern: {glossary:Term Name}
    # Replace with: [`Term Name`](../../references/glossary/#term-name)
    # Uses absolute-from-build-root path that works in Documenter's HTML output

    result = content
    for match in eachmatch(r"\{glossary:([^}]+)\}", content)
        term = match.captures[1]
        anchor = lowercase(replace(term, r"\s+" => "-"))
        # Link format: /references/glossary/#anchor (absolute path works from any page)
        # Documenter converts .md to .html in build output
        replacement = "[`$term`](/references/glossary/#$anchor)"
        result = replace(result, match.match => replacement; count=1)
    end
    return result
end

"""
    GlossaryLinkerHook()

Documenter hook that preprocesses markdown files to convert glossary references.
"""
struct GlossaryLinkerHook end

# Preprocess all markdown files in a directory
function preprocess_directory(dir_path::String)
    for (root, dirs, files) in walkdir(dir_path)
        for file in files
            if endswith(file, ".md")
                filepath = joinpath(root, file)
                content = read(filepath, String)
                modified_content = glossary_link_preprocessor(content)

                if modified_content != content
                    write(filepath, modified_content)
                end
            end
        end
    end
end

export glossary_link_preprocessor, GlossaryLinkerHook
