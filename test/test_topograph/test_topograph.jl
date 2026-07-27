"""
Tests for: the generalized topological graph layer (src/topograph/), split to mirror the source layout one file per src/topograph/*.jl.
"""

using Test
using Qritical

include("test_ids.jl")
include("test_layout.jl")
include("test_wire.jl")
include("test_leg.jl")
include("test_attachment.jl")
include("test_gengraph.jl")
include("test_ordinary_graph.jl")
include("test_orientation.jl")
include("test_network.jl")
include("test_pin.jl")
include("test_compactify.jl")
include("test_progressive.jl")
