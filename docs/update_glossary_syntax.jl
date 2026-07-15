#!/usr/bin/env julia
# Script to convert glossary links from [@ref syntax] to {glossary:term} syntax

function convert_glossary_links(filepath::String)
    content = read(filepath, String)

    # Pattern: [`term`](@ref Glossary#anchor)
    # Replace with: {glossary:term}

    # Convert both old @ref format and current {glossary:} format to new format
    modified = content

    # First pass: old @ref syntax
    for match in eachmatch(r"\[\`([^\`]+)\`\]\(@ref\s+Glossary#[^\)]+\)", content)
        term = match.captures[1]
        replacement = "{glossary:$term}"
        modified = replace(modified, match.match => replacement; count=1)
    end

    # Second pass: current {glossary:} syntax - will be converted by preprocessor
    # (this just ensures we have consistent syntax before preprocessing)

    if modified != content
        write(filepath, modified)
        println("✓ Updated: $filepath")
        return true
    else
        return false
    end
end

# Process all markdown files in src/api
api_dir = joinpath(@__DIR__, "src", "api")
updated = Int[]

for file in readdir(api_dir)
    if endswith(file, ".md")
        filepath = joinpath(api_dir, file)
        if convert_glossary_links(filepath)
            push!(updated, 1)
        end
    end
end

println("\nUpdated $(length(updated)) files with new glossary syntax")
