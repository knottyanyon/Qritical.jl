#!/usr/bin/env python3
"""
Convert Jupyter notebooks in docs/src/exercises to Literate.jl format (.jl files)
so they can be included in the Documenter build and rendered in the documentation.
"""

import json
import os
from pathlib import Path

def convert_notebooks_to_literate():
    exercises_dir = Path(__file__).parent.parent / "docs" / "src" / "exercises"

    # Find all ex_*.ipynb files
    for notebook_path in sorted(exercises_dir.glob("*/ex_*.ipynb")):
        exercise_name = notebook_path.stem  # e.g., "ex_01"
        output_path = notebook_path.parent / f"{exercise_name}.jl"

        print(f"Converting: {notebook_path.relative_to(exercises_dir.parent)} → {exercise_name}.jl")
        convert_notebook(notebook_path, output_path)

def convert_notebook(notebook_path, output_path):
    """Convert a Jupyter notebook to Literate.jl format."""

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    literate_lines = []

    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type")

        if cell_type == "code":
            # Code cell
            source = "".join(cell.get("source", []))

            # Skip setup lines
            lines = []
            for line in source.split("\n"):
                # Skip Pkg.activate and related setup
                if any(skip in line for skip in ["Pkg.activate", "@__DIR__", "import Pkg"]):
                    continue
                if line.strip():
                    lines.append(line)

            if lines:
                literate_lines.append("\n".join(lines))
                literate_lines.append("")

        elif cell_type == "markdown":
            # Markdown cell
            source = "".join(cell.get("source", []))

            # Convert to Literate format
            for line in source.split("\n"):
                stripped = line.strip()

                # Skip HTML comments
                if stripped.startswith("<!--"):
                    continue

                # Format headings and text
                if stripped.startswith("#"):
                    literate_lines.append(f"#md {line}")
                elif stripped:
                    literate_lines.append(f"#md {line}")
                else:
                    literate_lines.append("#md")

            literate_lines.append("")

    # Write output
    with open(output_path, "w") as f:
        f.write("\n".join(literate_lines))

if __name__ == "__main__":
    convert_notebooks_to_literate()
    print("✓ Conversion complete")
