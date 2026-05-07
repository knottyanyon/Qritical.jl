
# !!! question "Task 1.3 — SVD a state"
#     Perform an SVD on the state `psi.jls` given in Moodle. The format is a tensor of rank ten, dimensions $2^{10} = 1024$. Find the Schmidt rank needed if singular values below $10^{−6}$ are discarded.

# A workaround to ensure that the data can be read during local testing as well as pages deployment build

DATA_ROOT = normpath(joinpath(@__FILE__, ".."));
FPATH_PSI = normpath(joinpath(DATA_ROOT, "psi.jls"));

#--
using Qritical: QriticalUtils, factorize_with_svd, reshape_tensor_for_bipartition, get_schmidt_coefficients, get_entanglement_entropy

#--
tolerance = 1e-6;
ψ = QriticalUtils.load_state_from_file(FPATH_PSI);

# !!! subquestion
#     **A)** For a bipartition of the system after the first site

partition_1 = [(1,)]
ψ_reshaped_1 = reshape_tensor_for_bipartition(ψ, partition_1);

U_1, Σ_1, Vt_1 = factorize_with_svd(ψ_reshaped_1; discard_below_threshold=true, threshold=tolerance);

# !!! subquestion
#     **B)** For a bipartition of the system in the middle

partition_2 = [(1, 2, 3, 4, 5)];
ψ_reshaped_2 = reshape_tensor_for_bipartition(ψ, partition_2);

U_2, Σ_2, Vt_2 = factorize_with_svd(ψ_reshaped_2; discard_below_threshold=true, threshold=tolerance);
#--


# # Notes

# - A small Schmidt rank means that the state can be approximated with fewer terms. 

# - The Schmidt rank is higher in the case of middle partition because the entanglement is also higher compared to case 1. 



# Can I explain it with Monogamy of entanglement? More particles to share --> lower entanglement
# how can I test this hypothesis? I can simply calculate the entanglement entropy but how can I know for sure that this difference is indeed due to monogamy and not some other reason? How so I do a control? falsifiability? 

using CairoMakie

#--
function plot_entanglement_entropy(ψ; discard_below_threshold=true, threshold=1e-6, units=:bits)
    N = ndims(ψ)
    sites = 1:N-1

    entropies = map(sites) do i
        partition = [Tuple(1:i)]
        ψ_reshaped = reshape_tensor_for_bipartition(ψ, partition)
        λ = get_schmidt_coefficients(ψ_reshaped; discard_below_threshold=discard_below_threshold, threshold=threshold)
        S = get_entanglement_entropy(λ)
    end

    unit_label = "bits"

    fig = Figure(size=(600, 380), fontsize=13)
    ax = Axis(
        fig[1, 1],
        xlabel="partition boundary after site i",
        ylabel="S ($unit_label)",
        title="entanglement entropy across bipartitions",
        xticks=collect(sites),
    )

    lines!(ax, collect(sites), entropies; color=:teal, linewidth=2.2)
    scatter!(ax, collect(sites), entropies;
        color=:teal, markersize=10, strokewidth=0.5, strokecolor=:white)

    return fig, collect(entropies)
end


#--
fig_truncated, entropies_trunc = plot_entanglement_entropy(ψ; discard_below_threshold=true, threshold=tolerance)
save(normpath(joinpath(DATA_ROOT, "entropy_truncated.png")), fig_truncated)

