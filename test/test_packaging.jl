using Qritical: pkgdir as _pkgdir

const SRCDIR = joinpath(pkgdir(Qritical), "src")

@testset "src layout" begin
    # Top-level subdirectories
    for dir in ("tensors", "min_model_kit", "operators", "states", "algorithms", "studies", "utils")
        @test isdir(joinpath(SRCDIR, dir))
    end

    # Nested lattice directory inside min_model_kit
    @test isdir(joinpath(SRCDIR, "min_model_kit", "lattice"))

    # Exact file → folder mapping
    expected = [
        ("tensors",                  "tix.jl"),
        ("tensors",                  "multix.jl"),
        ("tensors",                  "partition.jl"),
        ("tensors",                  "qtensor.jl"),
        ("tensors",                  "spectrum.jl"),
        ("tensors",                  "svd.jl"),
        ("tensors",                  "storage_format.jl"),
        ("min_model_kit",            "dof.jl"),
        ("min_model_kit/lattice",    "geometry.jl"),
        ("min_model_kit/lattice",    "symmetries.jl"),
        ("operators",                "operator.jl"),
        ("operators",                "finite_mpo.jl"),
        ("operators",                "correlators.jl"),
        ("states",                   "mps.jl"),
        ("states",                   "canonicalize.jl"),
        ("states",                   "vidal.jl"),
        ("algorithms",               "power_method.jl"),
        ("algorithms",               "ed.jl"),
        ("algorithms",               "tebd.jl"),
        ("algorithms",               "quench.jl"),
        ("studies",                  "disorder.jl"),
        ("utils",                    "io.jl"),
    ]
    for (folder, file) in expected
        @test isfile(joinpath(SRCDIR, folder, file))
    end

    # src/ root contains exactly one .jl file
    root_jl = filter(f -> endswith(f, ".jl"), readdir(SRCDIR))
    @test root_jl == ["Qritical.jl"]
end

@testset "only the module root includes by path" begin
    for (root, _, files) in walkdir(SRCDIR)
        for file in files
            endswith(file, ".jl") || continue
            path = joinpath(root, file)
            path == joinpath(SRCDIR, "Qritical.jl") && continue
            text = read(path, String)
            for line in split(text, '\n')
                @test !occursin(r"^\s*include\(", line)
            end
        end
    end
end
