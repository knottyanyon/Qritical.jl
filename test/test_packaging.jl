using Qritical: pkgdir as _pkgdir

const SRCDIR = joinpath(pkgdir(Qritical), "src")

@testitem "src layout" begin
    # Top-level subdirectories
    for dir in (
        "core", "tensors", "operators", "states", "algorithms", "studies", "utils", "models"
    )
        @test isdir(joinpath(SRCDIR, dir))
    end

    @test !isdir(joinpath(SRCDIR, "experimental"))

    # Exact file → folder mapping
    expected = [
        ("core", "tix.jl"),
        ("tensors", "partition.jl"),
        ("tensors", "qtensor.jl"),
        ("tensors", "tensor_utils.jl"),
        ("tensors", "bond.jl"),
        ("tensors", "ortho_center.jl"),
        ("tensors", "spectrum.jl"),
        ("tensors", "svd.jl"),
        ("tensors", "storage_format.jl"),
        ("operators", "operator.jl"),
        ("operators", "finite_mpo.jl"),
        ("operators", "correlators.jl"),
        ("states", "mps.jl"),
        ("states", "canonicalize.jl"),
        ("states", "vidal.jl"),
        ("algorithms", "power_method.jl"),
        ("algorithms", "ed.jl"),
        ("algorithms", "tebd.jl"),
        ("studies", "study.jl"),
        ("studies", "evolution.jl"),
        ("studies", "disorder.jl"),
        ("utils", "io.jl"),
        ("utils", "deprecations.jl"),
        ("models", "models.jl"),
        ("models", "spacetime_dim.jl"),
        ("models", "algebra_tags.jl"),
        ("models", "dof.jl"),
    ]
    for (folder, file) in expected
        @test isfile(joinpath(SRCDIR, folder, file))
    end

    # src/ root contains exactly one .jl file
    root_jl = filter(f -> endswith(f, ".jl"), readdir(SRCDIR))
    @test root_jl == ["Qritical.jl"]
end

@testitem "only the module root includes by path" begin
    for (root, _, files) in walkdir(SRCDIR)
        for file in files
            endswith(file, ".jl") || continue
            path = joinpath(root, file)
            path == joinpath(SRCDIR, "Qritical.jl") && continue
            path == joinpath(SRCDIR, "core", "core.jl") && continue
            path == joinpath(SRCDIR, "models", "models.jl") && continue
            text = read(path, String)
            for line in split(text, '\n')
                @test !occursin(r"^\s*include\(", line)
            end
        end
    end
end
