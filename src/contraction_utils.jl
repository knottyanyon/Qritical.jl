
# The computational cost of multiplying a d1-by-d2 dimensional matrix A with a d2-by-d3 dimensional matrix B is: cost(A×B) = d1∙d2∙d3. Given the equivalence with matrix multiplication, this is also the cost of a binary tensor contraction (where each dimension d1, d2, d3 may now result as the product of several tensor indices from the reshapes).
# Another way of computing the cost of contracting A and B is to take the product of the total dimensions, denoted |dim(A)| and |dim(B)|, of each tensor divided by the total dimension of the contracted indices, denoted |dim(A∩B)|. Examples are given below:

# https://www.tensors.net/tutorial-1
# https://itensor.org/docs.cgi?page=tutorials/cost
# Helper functions to estimate the computational cost of doing a contraction (measured in number of floating-point scalar multiplications)

# The horizontal lines contain m terms (although in general, the left and right legs do not need to have the same number of terms). This number is called the bond dimension and can be thought of as the size of the wavefunction on each site in an MPS (meaning, the size of the local MPS matrix for that site). Specifically for DMRG, this number corresponds to the number of many body states kept in the Schmidt decomposition.

# The vertical leg corresponds to the physical index, d , and typically ranges over only a few values. The bond dimension may range over thousands of values.
function estimate_multiplication_cost(mat_A, mat_B)
    d1, d2_A = size(mat_A)
    d2_B, d3 = size(mat_B)
    if d2_A != d2_B
        error("matrix dimensions mismatch") # todo: write a better error message
    else
        cost = d1 * d2_A * d3
        println("$cost number of scalar multiplications needed.")
        return cost
    end
end

function estimate_contraction_cost(tensor_A, tensor_B)
    # todo
end