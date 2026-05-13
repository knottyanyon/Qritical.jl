
# # Task 1.2 — SVD a matrix

# !!! question "Task 1.2 — SVD a matrix"
#     Perform an SVD on the matrix $A$ given in Moodle as `A.txt` and find the Schmidt rank needed if singular values below $10^{−3}$ are discarded.

# A workaround to ensure that the data can be read during local testing as well as pages deployment build

DATA_ROOT = normpath(joinpath(@__FILE__, ".."));
FPATH_A = normpath(joinpath(DATA_ROOT, "A.txt"));
#--

#--
using DelimitedFiles

A_mat = readdlm(FPATH_A);


# Compute the singular value decomposition (SVD) of `A_mat` using the custom function [`factorize_with_svd`](@ref)

using Qritical: factorize_with_svd
#--

tolerance = 10E-3;
left_singular_mat, singular_mat, right_singular_mat = factorize_with_svd(A_mat; discard_below_threshold=true, threshold=tolerance);



# ## Notes

# Everything below is based on [trefethen_1997](@cite) and [golub_vanloan_2013](@cite).

# ### Matrix norms

# > “A norm on a vector space plays the same role as absolute value: it furnishes a distance measure.”
# >
# > — [golub_vanloan_2013](@cite)


# !!! definition "Induced matrix norm"
#     Given vector norms ``\|\cdot\|_{(n)}``  on the domain and ``\|\cdot\|_{(m)}`` on range of ``A \in \mathbb{C}^{m \times n}``, then we define ``\|A\|_{(m,n)}`` as the matrix norm induced by ``\|\cdot\|_{(m)}`` and ``\|\cdot\|_{(n)}`` as the smallest number ``C`` such that
#
#     ```math
#     \|Ax\|_{(m)} \leq C\|x\|_{(n)}, \quad \forall x \in \mathbb{C}^n.
#     ```
#     Put in simpler terms, ``C`` is the maximum factor by which ``A`` can "stretch" a vector ``x``. 
#


# - One can also define matrix norms that are not induced by the vector norms on its domain and range. We make use of the fact that an ``m \times n`` matrix can be viewed as a vector in an ``mn``-dimensional space with each of its ``mn`` matrix entries considered an independent coordinate. Using this interpretation we can define a general matrix norm in the``mn``-dimensional vector space of matrices similar to how the vector norm is defined by imposing conditions on the norm.


# !!! definition "General matrix norm"
#     A general matrix norm must satisfy the following conditions:
#
#     ```math
#     \begin{aligned}
#     &(1) \quad \|A\| \geq 0, \text{ and } \|A\| = 0 \text{ only if } A = 0, \\
#     &(2) \quad \|A + B\| \leq \|A\| + \|B\|, \\
#     &(3) \quad \|\alpha A\| = |\alpha| \|A\|, \quad \forall \alpha \in \mathbb{C}.
#     \end{aligned}
#     ```

# - The *Frobenius norm* (also called the *Hilbert-Schmidt norm*) is an example of a general matrix norm that is not induced by any vector norm. 

# !!! definition "Frobenius norm"
#     There are multiple ways to express the definition of the Frobenius norm ``\|A\|_F``.
#     - In terms of the matrix entries of ``A`` (it's evident from this form that it is the same as the *Euclidean 2-norm* of ``A`` viewed as an ``mn``- dimensional vector):
#
#     ```math
#     \|A\|_F = \left( \sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2 \right)^{1/2}.
#     ```
#     - In terms of the individual columns ``a_j`` (or rows ``a_i``) of ``A``:
#     ```math
#     \|A\|_F = \left( \sum_{j=1}^{n} {\|a_j\|_2}^2 \right)^{1/2}.
#     ```
#     - In terms of the trace:
#     ```math
#     \|A\|_F = \sqrt{\operatorname{tr}(A^*A)} = \sqrt{\operatorname{tr}(AA^*)}.
#     ```
#

# - Both the induced matrix 2-norm and Frobenius norm are invariant under multiplication by unitary matrices. 






# ### Singular Value Decomposition
#
# > “The SVD makes it possible for us to say that every matrix is diagonal— if only one uses the proper bases for the domain and range spaces.”
# >
# > — [trefethen_1997](@cite)
#

# | | Eigendecomposition | Singular Value Decomposition |
# |---|---|---|
# | **Bases** | One basis (the eigenvectors) | Two different bases (left and right singular vectors) |
# | **Orthogonality** | Basis generally not orthogonal | Orthonormal bases |
# | **Existence** | Not all square matrices have one | All matrices, including rectangular ones, have one |
# | **Scope of application** | problems involving the behavior of iterated forms of ``A`` such as ``A^k`` or ``e^{tA}``, | problems involving the behavior of ``A`` itself, or ``A^{-1}``|

# !!! theorem
#     ```math
#     \|A\|_2 = \sigma_1 \quad \text{and} \quad \|A\|_F = \sqrt{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2}.
#     ```


# !!! theorem 
#     The nonzero singular values of ``A`` are the square roots of the nonzero eigenvalues of ``A^*A`` or ``AA^*``. (These matrices have the same nonzero eigenvalues.)
# 

# !!! theorem 
#     If ``A=A^*``, then the singular values of ``A`` are the absolute values of the eigenvalues of ``A``.
# 


# !!! theorem "Schmidt, Mirsky, Eckhart- Young Theorem"
#      Given a matrix ``A`` its rank-``s`` truncated SVD approximation ``A_s`` is the best rank ``s`` approximation of ``A`` with respect to both the induced 2-norm and the Frobenius norm


# ### Low-rank matrix approximations in Frobenius norm

# !!! theorem "Matrix approximation as a sum of rank-one matrices"
#
#     ```math
#     A = \sum_{j=1}^{r} \sigma_j u_j v_j^*.
#     ```
#

# > ... represents a decomposition into rank-one matrices with a deeper property: the ``\nu^{th}`` partial sum captures as much of the energy of ``A`` as possible. This statement holds with "energy" defined by either the 2-norm or the Frobenius norm.
# >
# > — [trefethen_1997](@cite)
#

# !!! theorem "Best low rank approximation problem"
#     For any ``\nu`` with ``0 \leq \nu \leq r``, define
#
#     ```math
#     A_\nu = \sum_{j=1}^{\nu} \sigma_j u_j v_j^*;
#     ```
#
#     if ``\nu = p = \min\{m, n\}``, define ``\sigma_{\nu+1} = 0``. Then
#
#     ```math
#     \|A - A_\nu\|_2 = \inf_{\substack{B \in \mathbb{C}^{m \times n} \\ \operatorname{rank}(B) \leq \nu}} \|A - B\|_2 = \sigma_{\nu+1}.
#     ```
#
#     ```math
#     \|A - A_\nu\|_F = \inf_{\substack{B \in \mathbb{C}^{m \times n} \\ \operatorname{rank}(B) \leq \nu}} \|A - B\|_F = \sqrt{ \sigma_{\nu+1} + \cdots + \sigma_r^2}.
#     ```


# > Theorem 5.8 has a geometric interpretation. What is the best approximation of a hyperellipsoid by a line segment? Take the line segment to be the longest axis. What is the best approximation by a two-dimensional ellipsoid? Take the ellipsoid spanned by the longest and the second-longest axis. Continuing in this fashion, at each step we improve the approximation by adding into our approximation the largest axis of the hyperellipsoid not yet included. After ``r`` steps, we have captured all of ``A``.
# >
# > — [trefethen_1997](@cite)
#

# ## Error analysis

# !!! definition " Floating point rounding function"
#     Let ``\mathrm{fl} : \mathbb{R} \to \mathbf{F}`` be a function giving the closest floating point approximation to a real number, i.e. its *rounded* equivalent in the floating point system. For all ``x \in \mathbb{R}``, there exists ``\epsilon`` with ``|\epsilon| \leq \epsilon_{\mathrm{machine}}`` such that
#
#     ```math
#     \mathrm{fl}(x) = x(1 + \epsilon).
#     ```
#      That is, the difference between a real number and its closest floating point approximation is always smaller than ``\epsilon_{\mathrm{machine}}`` in relative terms.



# !!! definition "Floating point arithmetic"
#     Let ``x`` and ``y`` be arbitrary floating point numbers, that is, ``x, y \in \mathbf{F}``. Let ``*`` be one of the operations ``+``, ``-``, ``\times``, or ``\div``, and let ``\circledast`` be its floating point analogue. Then ``x \circledast y`` must be given exactly by
#
#     ```math
#     x \circledast y = \mathrm{fl}(x * y).
#     ```


# !!! definition "flops"
#     The number of floating point operations that an algorithm requires. Each addition, subtraction, multiplication, division, or square root counts as one *flop*.
#


# !!! theorem "Fundamental Axiom of Floating Point Arithmetic"
#     For all ``x, y \in \mathbf{F}``, there exists ``\epsilon`` with ``|\epsilon| \leq \epsilon_{\mathrm{machine}}`` such that
#
#     ```math
#     x \circledast y = (x * y)(1 + \epsilon).
#     ```
#     In words, every operation of floating point arithmetic is exact up to a relative error of size at most ``\epsilon_{\mathrm{machine}}``.

# - IEEE single precision arithmetic ``\epsilon_{\mathrm{machine}} \approx 10^{-8}`` 
# - IEEE double precision arithmetic ``\epsilon_{\mathrm{machine}} \approx 10^{-16}`` 

# > It is important to understand that ``\epsilon_{\mathrm{machine}}`` is not the smallest floating-point number that can be represented on a machine. That number depends on how many bits there are in the exponent, while ``\epsilon_{\mathrm{machine}}`` depends on how many bits there are in the mantissa 
# >
# > [press_2007a](@cite).
#

# > Pretty much any arithmetic operation among floating numbers should be thought of as introducing an additional fractional error of at least ``\epsilon_{\mathrm{machine}}``. This type of error is called **roundoff error** 
# >
# > [press_2007a](@cite).
#

# > Roundoff error is a characteristic of computer hardware. There is another, different, kind of error that is a characteristic of the program or algorithm used, independent of the hardware on which the program is executed. Many numerical algorithms compute “discrete” approximations to some desired “continuous” quantity. For example, an integral is evaluated numerically by computing a function at a discrete set of points, rather than at “every” point. Or, a function may be evaluated by summing a finite number of leading terms in its infinite series, rather than all infinity terms. In cases like this, there is an adjustable parameter, e.g., the number of points or of terms, such that the “true” answer is obtained only when that parameter goes to infinity. Any practical calculation is done with a finite, but sufficiently large, choice of that parameter. The discrepancy between the true answer and the answer obtained in a practical calculation is called the truncation error. Truncation error would persist even on a hypothetical, “perfect” computer that had an infinitely accurate representation and no roundoff error. Truncation error, on the other hand, is entirely under the programmer’s control. Most of the time, truncation error and roundoff error do not strongly interact with one another.
# >
# > A calculation can be imagined as having, first, the truncation error that it would have if run on an infinite-precision computer, “plus” the roundoff error associated with the number of operations performed.
# >
# > [press_2007a](@cite).