#!/bin/bash

# Script to add glossary links to API pages
# This script performs targeted replacements to add glossary cross-references

cd src/api

# spectrum.md
sed -i '' 's/\bSVD\b/[`SVD`](@ref Glossary#singular-value-decomposition)/g' spectrum.md
sed -i '' 's/\bcanonical form\b/[`canonical form`](@ref Glossary#canonical-form)/g' spectrum.md
sed -i '' 's/\bbond\b/[`bond`](@ref Glossary#bond)/g' spectrum.md
sed -i '' 's/\bEntanglement entropy\b/[`Entanglement entropy`](@ref Glossary#entanglement-entropy)/g' spectrum.md
sed -i '' 's/\bSchmidt rank\b/[`Schmidt rank`](@ref Glossary#schmidt-rank)/g' spectrum.md
sed -i '' 's/\bentanglement spectrum\b/[`entanglement spectrum`](@ref Glossary#entanglement-spectrum)/g' spectrum.md
sed -i '' 's/\bspectral gap\b/[`spectral gap`](@ref Glossary#spectral-gap)/g' spectrum.md
sed -i '' 's/\borthogonality centre\b/[`orthogonality centre`](@ref Glossary#orthogonality-centre)/g' spectrum.md
sed -i '' 's/\bMPS\b/[`MPS`](@ref Glossary#matrix-product-state)/g' spectrum.md

# mps.md
sed -i '' 's/\bMPS\b/[`MPS`](@ref Glossary#matrix-product-state)/g' mps.md
sed -i '' 's/\bbond dimension\b/[`bond dimension`](@ref Glossary#bond-dimension)/g' mps.md
sed -i '' 's/\bcanonical form\b/[`canonical form`](@ref Glossary#canonical-form)/g' mps.md
sed -i '' 's/\bcanonical\b/[`canonical`](@ref Glossary#canonical-form)/g' mps.md
sed -i '' 's/\btruncation\b/[`truncation`](@ref Glossary#truncation)/g' mps.md
sed -i '' 's/\bGauge\b/[`Gauge`](@ref Glossary#gauge)/g' mps.md
sed -i '' 's/\bIsometry\b/[`Isometry`](@ref Glossary#isometry)/g' mps.md

# dof.md
sed -i '' 's/\bDoF\b/[`DoF`](@ref Glossary#degree-of-freedom)/g' dof.md
sed -i '' 's/\bCCR\b/[`CCR`](@ref Glossary#ccr)/g' dof.md
sed -i '' 's/\bCAR\b/[`CAR`](@ref Glossary#car)/g' dof.md

# operator.md
sed -i '' 's/\bHamiltonian\b/[`Hamiltonian`](@ref Glossary#hamiltonian)/g' operator.md

# mpo.md
sed -i '' 's/\bMPO\b/[`MPO`](@ref Glossary#matrix-product-operator)/g' mpo.md
sed -i '' 's/\bMatrix Product Operator\b/[`Matrix Product Operator`](@ref Glossary#matrix-product-operator)/g' mpo.md

# tebd.md
sed -i '' 's/\bTEBD\b/[`TEBD`](@ref Glossary#tebd)/g' tebd.md
sed -i '' 's/\bTime-Evolving Block Decimation\b/[`Time-Evolving Block Decimation`](@ref Glossary#tebd)/g' tebd.md
sed -i '' 's/\bTrotter\b/[`Trotter`](@ref Glossary#tebd)/g' tebd.md
sed -i '' 's/\btruncation\b/[`truncation`](@ref Glossary#truncation)/g' tebd.md

# quench.md
sed -i '' 's/\bquench\b/[`quench`](@ref Glossary#quench)/g' quench.md
sed -i '' 's/\bTEBD\b/[`TEBD`](@ref Glossary#tebd)/g' quench.md
sed -i '' 's/\bproduct state\b/[`product state`](@ref Glossary#product-state)/g' quench.md

# ed.md
sed -i '' 's/\bLanczos\b/[`Lanczos`](@ref Glossary#lanczos)/g' ed.md
sed -i '' 's/\bspectral gap\b/[`spectral gap`](@ref Glossary#spectral-gap)/g' ed.md

# ed_time.md
sed -i '' 's/\bspectral gap\b/[`spectral gap`](@ref Glossary#spectral-gap)/g' ed_time.md

# power_method.md
sed -i '' 's/\bspectral gap\b/[`spectral gap`](@ref Glossary#spectral-gap)/g' power_method.md

# disorder.md
sed -i '' 's/\bMBL\b/[`MBL`](@ref Glossary#many-body-localization)/g' disorder.md
sed -i '' 's/\barea law\b/[`area law`](@ref Glossary#area-law)/g' disorder.md

# io.md
sed -i '' 's/\bSchmidt\b/[`Schmidt`](@ref Glossary#schmidt-decomposition)/g' io.md

echo "Glossary links added to API pages"
