# # 1. SVD a state

# Perform an SVD on the state `psi.jls` given in Moodle. The format is a tensor of rank 10, dimensions $2^{1]0} = 1024$.
# Find the Schmidt rank needed if singular values below $10^{−6}$ are discarded for:

# - (a) a bipartition of the system after the first site
# - (b) a bipartition of the system in the middle

using Qritical: QriticalUtils, factorize_with_svd, reshape_tensor_for_bipartition, validate_bipartition_indices
#--

DATA_ROOT = normpath(joinpath(@__FILE__, "..")) ## A workaround to ensure that the data can be read during local testing as well as pages deployment build


FPATH_PSI = normpath(joinpath(DATA_ROOT, "psi.jls"))

tolerance = 1e-6

#--
ψ = QriticalUtils.load_state_from_file(FPATH_PSI);

#--
partition_1 = [(1,)]
ψ_reshaped_1 = reshape_tensor_for_bipartition(ψ, partition_1);

factorize_with_svd(ψ_reshaped_1; discard_below_threshold=true, threshold=tolerance);

# A small Schmidt rank means that the state can be approximated with fewer terms. 
#--

partition_2 = [(1, 2, 3, 4, 5)];
ψ_reshaped_2 = reshape_tensor_for_bipartition(ψ, partition_2);

factorize_with_svd(ψ_reshaped_2; discard_below_threshold=true, threshold=tolerance);

# The Schmidt rank is higher in the case of middle partition because the entanglement is also higher compared to case 1. 

# Can I explain it with Monogamy of entanglement? More particles to share --> lower entanglement
# how can I test this hypothesis? I can simply calculate the entanglement entropy but how can I know for sure that this difference is indeed due to monogamy and not some other reason? How so I do a control? falsifiability? 