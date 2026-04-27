using CairoMakie

"""
    setup_size_N_rand_input(N::Int)

Generate random input matrices for benchmarking matrix multiplication.

# Arguments
- `N::Int`: The size of the square matrices.

# Returns
- A tuple `(A, B, C)` where `A` and `B` are random `N x N` matrices, and `C` is an uninitialized `N x N` matrix.
"""
function setup_size_N_rand_input(N::Int)
    A = rand(N, N)
    B = rand(N, N)
    C = Matrix{Float64}(undef, N, N)
    return A, B, C
end

"""
    contract_N_ijk(N; A, B, C)

Perform matrix multiplication C = A * B using the ijk loop order.

# Arguments
- `N::Int`: The size of the square matrices.
- `A`: The first input matrix.
- `B`: The second input matrix.
- `C`: The output matrix, modified in place.

# Returns
- The matrix `C` containing the result of A * B.
"""
function contract_N_ijk(N; A, B, C)
    for i in 1:N
        for j in 1:N
            C[i, j] = 0.0 # initialize the element to zero before summing over k
            for k in 1:N
                C[i, j] += A[i, k] * B[k, j]
            end
        end
    end
    return C
end
