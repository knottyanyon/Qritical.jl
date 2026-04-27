# ext/hinton_recipe.jl

@recipe(Hinton, matrix) do scene
    Attributes(;
        colormap=:twilight,
        colorrange=(-π, π),
        marker=Rect3f(Vec3f(-0.5, -0.5, 0), Vec3f(1, 1, 0)),
        scale=0.85,
        grid_color=(:black, 0.02),
        show_grid=true,
    )
end

function Makie.plot!(p::Hinton{<:Tuple{AbstractMatrix{<:Number}}})
    mat = p[:matrix]

    plot_data = lift(mat, p.scale) do m, s
        max_val = maximum(abs.(m))
        div_val = max_val == 0 ? 1.0 : max_val

        len = length(m)
        pos = Vector{Point3f}(undef, len)
        m_sizes = Vector{Vec3f}(undef, len)
        colors = Vector{Float32}(undef, len)

        # Use Cartesian pairs for robust mapping
        for (idx, val) in pairs(IndexCartesian(), m)
            i, j = idx.I
            l_idx = LinearIndices(m)[idx]

            # Map: Row (i) -> Y, Column (j) -> X
            pos[l_idx] = Point3f(j, i, 0)

            # Magnitude -> Size
            mag = (abs(val) / div_val) * s
            m_sizes[l_idx] = Vec3f(mag, mag, 1)

            # Phase -> Color
            colors[l_idx] = Float32(angle(val))
        end
        return pos, m_sizes, colors
    end

    # Background Grid
    if p.show_grid[]
        grid_pos = lift(mat) do m
            [Point3f(idx[2], idx[1], -0.1) for idx in keys(IndexCartesian(), m)][:]
        end

        meshscatter!(
            p,
            grid_pos;
            marker=p.marker,
            markersize=1.0,
            color=p.grid_color,
            shading=NoShading,
        )
    end

    # Matrix data
    meshscatter!(
        p,
        lift(d -> d[1], plot_data);
        markersize=lift(d -> d[2], plot_data),
        color=lift(d -> d[3], plot_data),
        marker=p.marker,
        colormap=p.colormap,
        colorrange=p.colorrange,
        shading=NoShading,
    )

    return p
end

function draw_complex_hinton(matrix::AbstractMatrix)
    fig = Figure()

    # yreversed so that it looks similar to the matrix row and column ordering
    ax = Axis(fig[1, 1]; aspect=DataAspect(), yreversed=true)

    hp = hinton!(ax, matrix)

    Colorbar(
        fig[1, 2],
        hp;
        label="Phase (rad)",
        ticks=((-π):(π / 2):π, [L"-π", L"-π/2", L"0", L"π/2", L"π"]),
    )
    hidedecorations!(ax)
    return fig
end

function draw_svd_hinton(M, svd_out)
    U, S, Vt = svd_out
    Σ = Diagonal(S)

    fig = Figure()
    # fig = Figure(size=(800, 300))
    grid_M = fig[1, 1] = GridLayout()
    grid_SVD = fig[1, 2] = GridLayout()
    grid_cbar = fig[1, 3] = GridLayout()

    ax_M = Axis(
        grid_M[1, 1];
        title=L"M",
        aspect=DataAspect(),
        yreversed=true,
        # limits=(0.5, size(M)[1] + 0.5, 0.5, size(M)[2] + 0.5),
    )

    ax_U = Axis(
        grid_SVD[1, 1];
        title=L"U",
        aspect=DataAspect(),
        yreversed=true,
        # limits=(0.5, size(U)[1] + 0.5, 0.5, size(U)[2] + 0.5),
    )

    ax_Σ = Axis(
        grid_SVD[1, 2];
        title=L"Σ",
        aspect=DataAspect(),
        yreversed=true,
        # limits=(0.5, size(Σ)[1] + 0.5, 0.5, size(Σ)[2] + 0.5),
    )

    ax_Vt = Axis(
        grid_SVD[1, 3];
        title=L"Vt",
        aspect=DataAspect(),
        yreversed=true,
        # limits=(0.5, size(Vt)[1] + 0.5, 0.5, size(Vt)[2] + 0.5),
    )

    hinton_M = hinton!(ax_M, M)
    hinton_U = hinton!(ax_U, U)
    hinton_Σ = hinton!(ax_Σ, Σ)
    hinton_Vt = hinton!(ax_Vt, Vt)

    # 4. Add the Colorbar for the phase
    # We pass 'hp' to the Colorbar so it knows the colormap and range
    Colorbar(
        grid_cbar[1, 1],
        hinton_M;
        label="Phase (radians)",
        ticks=((-π):(π / 2):π, ["-π", "-π/2", "0", "π/2", "π"]),
        # label="Phase",
    )

    hidedecorations!(ax_M)
    hidedecorations!(ax_U)
    hidedecorations!(ax_Σ)
    hidedecorations!(ax_Vt)

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colsize!(fig.layout, 2, Aspect(1, 3.0))

    # rowsize!(fig.layout, 1, Relative(2 / 3))
    resize_to_layout!(fig)
    trim!(fig.layout)
    return fig
end