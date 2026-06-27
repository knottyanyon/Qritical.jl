# §2 Geometry layer — lattice structures for Hamiltonian construction.
# A geometry answers two queries — sites and bonds — which is all the
# Hamiltonian builder consumes.  Deferred: Square, Torus, general Lattice{V,E}.

struct Chain <: AbstractGeometry
    L::Int
    periodic::Bool
end
Chain(L::Int) = Chain(L, false)

sites(g::Chain) = 1:g.L

function bonds(g::Chain)
    L = g.L
    if g.periodic
        [(i, mod1(i + 1, L)) for i in 1:L]
    else
        [(i, i + 1) for i in 1:L-1]
    end
end
