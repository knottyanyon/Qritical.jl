using LinearAlgebra
using DelimitedFiles
# using TensorOperations

A = readlm("A.txt")
dimension = size(A, 1)

F = svd(A)
U = F.U
S = F.S
Vt = F.Vt

tol = 1e-3
keep = S .> tol

Schmidt_rank = count(keep)

S = Diagonal(S)

Utrunc = U[:, keep]
Strunc = S[keep, keep]
Vttrunc = Vt[keep, :]

B_trunc = Utrunc * Strunc * Vttrunc

# @tensor B_trunc2[i,j] := U[i,k] * S[k,l] * Vt[l,j]

