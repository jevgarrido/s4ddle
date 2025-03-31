#![allow(non_snake_case)]

use crate::operators::*;
use crate::plugins::*;

/// Implementation of the preconditioned MINRES algorithm.
/// Coefficient matrix and preconditioner are NOT explicittly required.
/// Matrix-vector products are outsourced to you through a "black box" interface.
pub fn minres_black_box_precond<'a, T: Float>(
    x: &'a mut [T],                      // Solution vector (initial guess)
    black_box: &impl BlackBoxPrecond<T>, // Black box plugin
    b: &[T],                             // Right-hand-side of the system
    shift: &T,                           // M ( A + shift * I ) x = M b, x = M^T y
    tolerance: &T,                       // Absolute tolerance
    maxiters: &usize,                    // maxium number of iterations allowed
    plugin: &mut impl Plugin<T>,         // User-defined interactive Plugin
    Av: &mut [T],
    Mv: &mut [T],
    mut v: &'a mut [T],
    mut v_next: &'a mut [T],
    mut p_prev: &'a mut [T],
    mut p: &'a mut [T],
) -> (isize, T, usize) {
    // Plug-in call ------------------------------
    plugin.start();
    // -------------------------------------------
    //
    // Initialization ----------------------------
    let mut success: isize = 0;
    let mut iters: usize = 0;

    let mut ta: T;
    let mut tb: T = T::zero();

    let mut X: T;
    let mut Y: T = T::zero();

    let mut cos: T = -T::one();
    let mut sin: T = T::zero();

    let mut ua: T;
    let mut ub: T;
    let mut uc: T = T::zero();
    let mut uc_next: T;

    let mut alpha: T;
    let mut residual: T;
    // -------------------------------------------
    //
    // Algorithm Initialization ------------------

    // v1 = r0 = M * [ b - ( A*x0 + shift * x0 ) ]
    Av.reset();
    black_box.apply_A(Av, x);
    Av.add(*shift, x);
    v_next.linear_comb(T::one(), b, -T::one(), Av);

    v.reset();
    black_box.apply_precond(v, v_next);

    residual = v.norm_2();

    if residual <= *tolerance {
        success = 1;
        plugin.end();
        return (success, residual, iters);
    }

    v.scale(T::one() / residual);

    p.copy_from_slice(&x);

    x.reset();
    black_box.apply_precond_inverse_transpose(x, p);

    v_next.reset();
    p_prev.reset();
    p.reset();

    Av.reset();
    Mv.reset();

    for kk in 1..=*maxiters {
        iters = kk;
        // Lanczos Iteration ---------------------
        Mv.reset();
        black_box.apply_precond_transpose(Mv, v);

        Av.reset();
        black_box.apply_A(Av, Mv);
        Av.add(*shift, Mv);
        ta = Mv.dot(Av);

        v_next.scale(-tb);
        v_next.add(-ta, v);

        Mv.reset();
        black_box.apply_precond(Mv, Av);
        v_next.add(T::one(), Mv);

        tb = v_next.norm_2();
        v_next.scale(T::one() / tb);
        // ---------------------------------------
        //
        // Apply Old Givens Rotation -------------
        ub = Y * cos + ta * sin;
        uc_next = tb * sin;

        X = Y * sin - ta * cos;
        Y = -tb * cos;
        // ---------------------------------------
        //
        // Compute New Givens Rotation -----------
        ua = X.hypot(tb);
        cos = X / ua;
        sin = tb / ua;
        // ---------------------------------------
        //
        // Compute Step --------------------------
        p.scale_add(-uc / ua, -ub / ua, p_prev);
        p.add(T::one() / ua, v);

        alpha = residual * cos;
        // ---------------------------------------
        //
        // Update Solution -----------------------
        x.add(alpha, p);
        residual *= sin;

        plugin.peek(&iters, x, p, &alpha, &residual);

        if residual <= *tolerance {
            success = 1;
            break;
        }
        // ---------------------------------------
        //
        // Close the loop ------------------------
        core::mem::swap(&mut v, &mut v_next);
        core::mem::swap(&mut p_prev, &mut p);
        uc = uc_next;
        // ---------------------------------------
    }

    p.copy_from_slice(&x);
    x.reset();
    black_box.apply_precond_transpose(x, p);

    plugin.end();
    return (success, residual, iters);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readwrite::*;
    use crate::precond_methods::*;
    use rstest::*;

    const TOLERANCE: f64 = 1e-12;
    const REDUCED_TOL: f64 = 1e-8;

    struct MatrixAndDiagonalPreconditioner<'a, T: Float> {
        rows: &'a [usize],
        cols: &'a [usize],
        vals: &'a [T],
        p_vals: &'a [T],
    }

    impl<T: Float> BlackBox<T> for MatrixAndDiagonalPreconditioner<'_, T> {
        #[inline(always)]
        fn apply_A(&self, product: &mut [T], vector: &[T]) {
            product.spmv(self.rows, self.cols, self.vals, vector);
        }
    }

    impl<T: Float> BlackBoxPrecond<T> for MatrixAndDiagonalPreconditioner<'_, T> {
        #[inline(always)]
        fn apply_precond(&self, product: &mut [T], vector: &[T]) {
            for (idx, val) in self.p_vals.iter().enumerate() {
                product[idx] += *val * vector[idx];
            }
        }
        #[inline(always)]
        fn apply_precond_transpose(&self, product: &mut [T], vector: &[T]) {
            self.apply_precond(product, vector);
        }
        #[inline(always)]
        fn apply_precond_inverse_transpose(&self, product: &mut [T], vector: &[T]) {
            for (idx, val) in self.p_vals.iter().enumerate() {
                product[idx] += (T::one() / *val) * vector[idx];
            }
        }
    }

    #[rstest]
    #[case(0.0, -3.0)]
    #[case(-3.0, -3.0)]
    #[case(-3.0, 5.0)]
    fn identity(#[case] x_vals: f64, #[case] b_vals: f64) {
        const N: usize = 10;

        let Arows: Vec<usize> = (0..N).collect();
        let Acols: Vec<usize> = (0..N).collect();
        let Avals: Vec<f64> = vec![1.0; N];
        let Pvals: Vec<f64> = vec![1.0; N]; // Identity preconditioner

        let b = vec![b_vals; N];

        let (mut x, mut error) = (vec![x_vals; N], vec![0.0; N]);

        let mut plugin = DoNothing::new();

        let black_box_instance =
            MatrixAndDiagonalPreconditioner { rows: &Arows, cols: &Acols, vals: &Avals, p_vals: &Pvals };

        let (_success, _residual, _iters) = minres_black_box_precond(
            &mut x,
            &black_box_instance,
            &b,
            &0.0,
            &TOLERANCE,
            &N,
            &mut plugin,
            &mut vec![0.0; N],
            &mut vec![0.0; N],
            &mut vec![0.0; N],
            &mut vec![0.0; N],
            &mut vec![0.0; N],
            &mut vec![0.0; N],
        );

        // Compute true residual
        error.spmv(&Arows, &Acols, &Avals, &x);
        let true_residual = error.scale_add(-1.0, 1.0, &b).norm_2();

        // Compute solution error
        let solution_error = error.linear_comb(1.0, &x, -1.0, &b).norm_2();

        assert!(true_residual <= TOLERANCE);
        assert!(solution_error <= REDUCED_TOL);
    }

    #[rstest]
    fn test_1by1() {
        let maxiters: usize = 100;
        let mut plugin = DoNothing::new();
        const SIZE: usize = 1;

        let Arows: [usize; 1] = [0];
        let Acols: [usize; 1] = [0];
        let Avals: [f64; 1] = [2.0; 1];
        let Pvals: [f64; 1] = [1.0 / (2.0f64.sqrt()); 1];

        let black_box_instance =
            MatrixAndDiagonalPreconditioner { rows: &Arows, cols: &Acols, vals: &Avals, p_vals: &Pvals };

        let sol: [f64; SIZE] = [1.0];

        let mut b: [f64; SIZE] = [0.0; SIZE];
        b.spmv(&Arows, &Acols, &Avals, &sol);

        let (mut x, mut error) = (vec![0.0; SIZE], vec![0.0; SIZE]);

        let (_success, _residual, _iters) = minres_black_box_precond(
            &mut x,
            &black_box_instance,
            &b,
            &0.0,
            &TOLERANCE,
            &maxiters,
            &mut plugin,
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
        );

        // Compute true residual
        error.spmv(&Arows, &Acols, &Avals, &x);
        let true_residual = error.scale_add(-1.0, 1.0, &b).norm_2();

        // Compute solution error
        let solution_error = error.linear_comb(1.0, &x, -1.0, &sol).norm_2();

        assert!(true_residual <= TOLERANCE);
        assert!(solution_error <= REDUCED_TOL);
    }

    #[rstest]
    fn test_2by2() {
        let maxiters: usize = 100;
        let mut plugin = DoNothing::new();
        const SIZE: usize = 2;

        let Arows: [usize; 3] = [0, 1, 0];
        let Acols: [usize; 3] = [0, 0, 1];
        let Avals: [f64; 3] = [1.0, 1.0, 1.0];
        let Pvals: [f64; 2] = [0.8408964, 1.0];

        let black_box_instance =
            MatrixAndDiagonalPreconditioner { rows: &Arows, cols: &Acols, vals: &Avals, p_vals: &Pvals };

        let sol: [f64; SIZE] = [1.0, 1.0];

        let mut b: [f64; SIZE] = [0.0; SIZE];
        b.spmv(&Arows, &Acols, &Avals, &sol);

        let (mut x, mut error) = (vec![0.0; SIZE], vec![0.0; SIZE]);

        let (_success, _residual, _iters) = minres_black_box_precond(
            &mut x,
            &black_box_instance,
            &b,
            &0.0,
            &TOLERANCE,
            &maxiters,
            &mut plugin,
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
        );

        // Compute true residual
        error.spmv(&Arows, &Acols, &Avals, &x);
        let true_residual = error.scale_add(-1.0, 1.0, &b).norm_2();

        // Compute solution error
        let solution_error = error.linear_comb(1.0, &x, -1.0, &sol).norm_2();

        assert!(true_residual <= TOLERANCE);
        assert!(solution_error <= REDUCED_TOL);
    }

    #[rstest]
    fn test_4by4() {
        const SIZE: usize = 4;
        let maxiters: usize = 2 * SIZE;
        let mut plugin = DoNothing::new();

        let Arows: [usize; 6] = [0, 2, 1, 3, 0, 1];
        let Acols: [usize; 6] = [0, 0, 1, 1, 2, 3];
        let Avals: [f64; 6] = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut Pvals: [f64; 4] = [0.0; 4];

        ruiz_scaling(&mut Pvals, &Arows, &Acols, &Avals, &10, &1e-12, &mut vec![0.0; 4]);

        let black_box_instance =
            MatrixAndDiagonalPreconditioner { rows: &Arows, cols: &Acols, vals: &Avals, p_vals: &Pvals };

        let sol: [f64; SIZE] = [1.0; SIZE];

        let mut b: [f64; SIZE] = [0.0; SIZE];
        b.spmv(&Arows, &Acols, &Avals, &sol);

        let (mut x, mut error) = (vec![0.0; SIZE], vec![0.0; SIZE]);

        let (_success, _residual, _iters) = minres_black_box_precond(
            &mut x,
            &black_box_instance,
            &b,
            &0.0,
            &TOLERANCE,
            &maxiters,
            &mut plugin,
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
            &mut vec![0.0; SIZE],
        );

        // Compute true residual
        error.spmv(&Arows, &Acols, &Avals, &x);
        let true_residual = error.scale_add(-1.0, 1.0, &b).norm_2();

        // Compute solution error
        let solution_error = error.linear_comb(1.0, &x, -1.0, &sol).norm_2();

        assert!(true_residual <= TOLERANCE);
        assert!(solution_error <= REDUCED_TOL);
    }

    #[rstest]
    #[case("./data/nemeth01.csv")]
    #[case("./data/nemeth26.csv")]
    #[case("./data/GHS_indef_qpband.csv")]
    #[case("./data/GHS_indef_tuma2.csv")]
    #[case("./data/GHS_indef_linverse.csv")]
    #[case("./data/FIDAP_ex4.csv")]
    fn test_matrix_market(#[case] file_path: &str) {
        let (Arows, Acols, Avals, size, _nonzeros) = mm_read(file_path);

        let mut Pvals: Vec<f64> = vec![0.0; size];

        ruiz_scaling(&mut Pvals, &Arows, &Acols, &Avals, &10, &1e-12, &mut vec![0.0; size]);

        let black_box_instance =
            MatrixAndDiagonalPreconditioner { rows: &Arows, cols: &Acols, vals: &Avals, p_vals: &Pvals };

        let mut plugin = DoNothing::new();
        let mut plug_fancy_donothing = DoNothingDecorator::extend(&mut plugin);
        let mut plug_extra_layer = DoNothingDecorator::extend(&mut plug_fancy_donothing);

        let mut x = vec![0.0; size];
        let mut sol: Vec<f64> = vec![1.0; size];
        let sol_norm: f64 = sol.norm_2();
        sol.scale(1.0 / sol_norm);

        let mut b: Vec<f64> = vec![0.0; size];
        b.spmv(&Arows, &Acols, &Avals, &sol);

        let mut error = vec![0.0; size];

        for shift in [0.0] {
            minres_black_box_precond(
                &mut x,
                &black_box_instance,
                &b,
                &shift,
                &TOLERANCE,
                &(2 * size),
                &mut plug_extra_layer,
                &mut vec![0.0; size],
                &mut vec![0.0; size],
                &mut vec![0.0; size],
                &mut vec![0.0; size],
                &mut vec![0.0; size],
                &mut vec![0.0; size],
            );
        }

        // Compute true residual
        error.spmv(&Arows, &Acols, &Avals, &x);
        let true_residual = error.scale_add(-1.0, 1.0, &b).norm_2();

        // Compute solution error
        let solution_error = error.linear_comb(1.0, &x, -1.0, &sol).norm_2();
        error.reset();

        assert!(true_residual <= TOLERANCE * 1e1);
        assert!(solution_error <= REDUCED_TOL);
    }
}
