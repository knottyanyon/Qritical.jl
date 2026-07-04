"""
    convert_notebooks_to_literate()

Convert Jupyter notebooks in docs/src/exercises to Literate.jl format (.jl files)
so they can be included in the Documenter build and rendered in the documentation.

Literate.jl uses a special format:
- Code cells: normal Julia code
- Markdown cells: `#md # Heading` or `#md Text`
- Output: `#md output` followed by the rendered output
"""

using JSON

function convert_notebooks_to_literate()
    exercises_dir = joinpath(@__DIR__, "..", "docs", "src", "exercises")

    # Find all ex_*.ipynb files
    for (root, dirs, files) in walkdir(exercises_dir)
        for file in files
            if startswith(file, "ex_") && endswith(file, ".ipynb")
                notebook_path = joinpath(root, file)
                exercise_name = splitext(file)[1]  # e.g., "ex_01"
                output_path = joinpath(root, "$(exercise_name).jl")

                println("Converting: $notebook_path → $output_path")
                convert_notebook(notebook_path, output_path)
            end
        end
    end
end

function convert_notebook(notebook_path::String, output_path::String)
    # Read notebook
    nb = JSON.parse(read(notebook_path, String))

    # Convert cells to Literate format
    literate_content = String[]

    for cell in nb["cells"]
        if cell["cell_type"] == "code"
            # Code cell
            source = join(cell["source"], "")

            # Skip Pkg.activate and other setup lines
            lines = split(source, "\n")
            filtered_lines = filter(lines) do line
                !contains(line, "Pkg.activate") &&
                !contains(line, "@__DIR__") &&
                !isempty(strip(line))
            end

            if !isempty(filtered_lines)
                push!(literate_content, join(filtered_lines, "\n"))
                push!(literate_content, "")
            end

        elseif cell["cell_type"] == "markdown"
            # Markdown cell
            source = join(cell["source"], "")

            # Convert to Literate format
            lines = split(source, "\n")
            for line in lines
                if !isempty(strip(line))
                    if startswith(strip(line), "#")
                        # It's a heading - keep the #
                        push!(literate_content, "#md $(line)")
                    elseif startswith(strip(line), "<!--")
                        # Skip HTML comments
                        continue
                    else
                        # Regular text
                        push!(literate_content, "#md $(line)")
                    end
                else
                    push!(literate_content, "#md")
                end
            end
            push!(literate_content, "")
        end
    end

    # Write to file
    open(output_path, "w") do f
        write(f, join(literate_content, "\n"))
    end
end

# Run the conversion
convert_notebooks_to_literate()
println("✓ Conversion complete")
