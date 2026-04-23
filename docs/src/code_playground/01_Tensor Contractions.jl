# # Understanding Indices and Contractions

# - [Tensors.net turotial 1](https://www.tensors.net/j-tutorial-1)
# - [ITensor examples](https://docs.itensor.org/ITensors/stable/examples/ITensor.html)

# Let's do this first with base Julia
using LinearAlgebra

## tensor with randomly generated entries
## order 3, dims: 2-by-3-by-4
A = rand(2, 3, 4);
display(A)

## identity matrix, order 2, dims: 5-by-5 (New syntax in Julia 0.7+)
B_int = Matrix{Int64}(I, 5, 5);
B_float = Matrix{Float64}(I, 5, 5);
display(B_int)
display(B_float)

## matrix of 0's, order 2, dims: 3-by-5
D = zeros(3, 5);
display(D)

## initialize complex random tensor
E = rand(2, 3) + im * rand(2, 3);
display(E)

## creating column vectors (default)
col_vec = [0.2, 0.5, 0.3];
rand_col_vec = rand(3);
display(col_vec)
display(rand_col_vec)

## creating row vectors
## notice the absence of commas between the entries
row_vec = [0.2 0.5 0.3];
display(row_vec)

## stacking two column vectors horizontally
stacked_hcol_vec = [col_vec rand_col_vec]; # we are creating a "row vector" of with elements col_vec and rand_col_vec
display(stacked_hcol_vec)
@show size(stacked_hcol_vec);
@show ndims(stacked_hcol_vec);
@show length(stacked_hcol_vec); ## number of entries in the vector

## stacking two column vectors vertically. Note the shape and dimensions of the result
stacked_vcol_vec = [col_vec, rand_col_vec];
display(stacked_vcol_vec)
@show size(stacked_vcol_vec);
@show ndims(stacked_vcol_vec);
@show length(stacked_vcol_vec); ## number of entries in the vector

## stacking the elements of two column vectors vertically. Note the shape and dimensions of the result
stacked_vcol_vec_el = [col_vec; rand_col_vec]; # the ; between the elements to stack the elements 
display(stacked_vcol_vec_el)
@show size(stacked_vcol_vec_el);
@show ndims(stacked_vcol_vec_el);
@show length(stacked_vcol_vec_el); ## number of entries in the vector

# Practice :  create Pauli matrices by specifying the entries going column-wise unlike Numpy 

sigma_x = [[0, 1.0] [1.0, 0]];
sigma_y = [[0.0, im] [-im, 0.0]];
sigma_z = [[1.0, 0] [0, -1.0]];

display(sigma_x)
display(sigma_y)
display(sigma_z)

## tensor of 1's, order 4, dims: 3 x 4 x 2 x 1
C = ones(3, 4, 2, 1);
display(C); ## notice how the tensor is displayed. it will print higher order tensors as slices across the dimensions

## Permute allows the index ordering of a tensor to be changed (but does not change the number of indices).
A_ijkl = rand(4, 4, 4, 4);
A_lijk = permutedims(A_ijkl, [4, 1, 2, 3]);
display(A_ijkl);
display(A_lijk);

## The reshape function which allows a collection of tensor indices to be combined into a single larger index (or vice-versa), thus can change the number of indices but not the total dimension.
B = rand(3, 3, 3);
Btilda = reshape(B, 3, 3^2);

# ## Contractions

# The usefulness of permute and reshape functions is that they allow a contraction between a pair of tensors (which we call a binary tensor contraction) to be recast as a matrix multiplication. Although the computational cost (measured in number of scalar multiplications) is the same both ways, it is usually preferable to recast as multiplication as modern hardware performs vectorized operations much faster than when using the equivalent FOR loop.

d = 3;
A = rand(d, d, d, d);
B = rand(d, d, d, d);

display(A);
display(B);

## Contraction example : C_ijkl = A_imjn * B_mkln

## Simple Julia
Ap = permutedims(A, [1, 3, 2, 4]); ## imjn -> ijmn
Bp = permutedims(B, [1, 4, 2, 3]); ## mkln -> mnkl
App = reshape(Ap, d^2, d^2);
Bpp = reshape(Bp, d^2, d^2);
Cpp = App * Bpp;
C = reshape(Cpp, d, d, d, d);
display(C)

## The same example but using ITensors
using ITensors
using ITensorUnicodePlots

## Use the Let block : code written in the Julia global scope can have some surprising behaviors. Putting your code into a let block avoids these issues
let
    ## open indices of A_imjn
    i = Index(d, "i")
    j = Index(d, "j")

    ## open indices of B_mkln
    k = Index(d, "k")
    l = Index(d, "l")

    ## contracted indices
    m = Index(d, "m")
    n = Index(d, "n")

    ## create the tensors with the appropriate labels
    A_ITensor = ITensor(A, i, m, j, n)
    B_ITensor = ITensor(B, m, k, l, n)

    ## Contract over shared indices m and n
    C_ITensor = @visualize A_ITensor * B_ITensor edge_labels = (tags=true,)

    ## if our contraction is correct, the resulting tensor C should have open indices i, j, k, l
    @show inds(C_ITensor)

    ## convert the C_ITensor to a regular Julia array and check that it matches the result from simple Julia
    C_ITensor_arr = Array(C_ITensor, i, j, k, l)

    @show (C_ITensor_arr ≈ C)
end
