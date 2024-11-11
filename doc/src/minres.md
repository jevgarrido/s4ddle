# Minimal Residual Algorithm - MINRES

This method solves linear system in the following forms:
$$ (A - s I ) x = b $$
$$ M (A - s I ) x = M b \quad x = M^\top y$$ 

Where \\(A\\) is a Real symmetric coefficient matrix (no assumption on its definiteness),
\\(s\\) is a Real scalar value,
\\(I\\) is the identity matrix,
\\(b\\) is an arbitrary Real vector,
\\(M\\) is a Real full-rank square preconditioning matrix (no assumption on its symmetry).

Internally, the preconditioned case is treated as follows:
$$ M (A - s I ) M^\top y = M b \, \quad x = M^\top y $$
which constitutes a symmetric system in terms of \\(y\\).


<!-- x = x_n + x_e0 + x_ei = x_n + sum_i (vi * x_e) vi --> 

<!-- x_e = x - x_n = sum_i (vi * x_e) vi = sum_i (vi * x) vi --> 

<!-- (vi * x) vi = (vi * x_e) vi -->



<!-- x = a1 e1 + a2 e2 + a3 e3 + ... + ai ei -->
<!-- x * ei = a1 (e1 * ei) + a2 (e2 * ei) + a3 (e3 * ei) + ... + ai (ei * ei) -->

<!-- ai (ei * ei) = x * ei - a1 (e1 * ei) - a2 (e2 * ei) - a3 (e3 * ei) - ... -->

<!-- ai (ei * ei) = x * ei - sum_{j\<i} aj (ej * ei) -->


## References:
* C. C. Paige and M. A. Saunders (1975). "Solution of sparse indefinite systems of linear equations", SIAM J. Numerical Analysis 12, 617-629
