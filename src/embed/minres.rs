#![allow(non_snake_case)]

use crate::embed::operators::*;

pub fn minres<'a, T: Float>(
    x: &mut [T],                 // Solution vector (initial guess)
    mrows: &[usize],             // Left-hand-side matrix: row indices
    mcols: &[usize],             // Left-hand-side matrix: column indices
    mvals: &[T],                 // Left-hand-side matrix: values
    b: &[T],                     // Right-hand-side of the system
    shift: &T,                   // ( A - shift * I ) x = b
    tolerance: &T,               // Absolute tolerance
    maxiters: &usize,            // maxium number of iterations allowed
    plugin: &mut impl Plugin<T>, // User-defined interactive Plugin
    Av: &mut [T],
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
    spmv(Av, mrows, mcols, mvals, x);
    Av.add(-*shift, x);
    v.linear_comb(T::one(), b, T::one().neg(), Av);

    residual = dot(v, v).sqrt();

    if residual <= *tolerance {
        plugin.end();
        return (1, residual, 0);
    }

    v.scale(T::one() / residual);

    v_next.reset();
    p_prev.reset();
    p.reset();

    for kk in 1..=*maxiters {
        iters = kk;
        // Lanczos Iteration ---------------------
        spmv(Av, mrows, mcols, mvals, v);
        Av.add(-*shift, v);
        ta = dot(v, Av);

        v_next.scale(-tb);
        v_next.add(T::one(), Av);
        v_next.add(-ta, v);

        tb = dot(v_next, v_next).sqrt();
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
        residual = residual * sin;

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

    plugin.end();
    return (success, residual, iters);
}

pub fn minres_precond() {
    todo!()
}

pub fn minres_black_box_precond() {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readwrite::*;
    use rstest::*;

    const TOLERANCE: f64 = 1e-12;
    const REDUCED_TOL: f64 = 1e-8;

    #[rstest]
    #[case(0.0, -3.0)]
    #[case(-3.0, -3.0)]
    #[case(-3.0, 5.0)]
    fn minres_identity(#[case] x_vals: f64, #[case] b_vals: f64) {
        const N: usize = 10;

        let Arows_cols: Vec<usize> = (0..N).collect();
        let Avals = vec![1.0; N];
        let b = vec![b_vals; N];

        let (mut x, mut error) = (vec![x_vals; N], vec![0.0; N]);
        let mut plugin = StopWatchAndPrinter::new();

        let (_success, _residual, _iters) = minres(
            &mut x,
            &Arows_cols,
            &Arows_cols,
            &Avals,
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
        );

        spmv(&mut error, &Arows_cols, &Arows_cols, &Avals, &x);
        error.add(-1.0, &b);

        let true_residual = dot(&error, &error).sqrt();

        error.linear_comb(1.0, &x, -1.0, &b);
        let solution_error = dot(&error, &error).sqrt();

        assert!(true_residual <= TOLERANCE);
        assert!(solution_error <= REDUCED_TOL);
    }

    #[rstest]
    fn minres_1by1() {
        let maxiters: usize = 100;
        let mut plugin = StopWatchAndPrinter::new();
        const SIZE: usize = 1;

        let Arows: [usize; 1] = [0];
        let Acols: [usize; 1] = [0];
        let Avals: [f64; 1] = [2.0; 1];

        let sol: [f64; SIZE] = [1.0];

        let mut b: [f64; SIZE] = [0.0; SIZE];
        spmv(&mut b, &Arows, &Acols, &Avals, &sol);

        let (mut x, mut error) = (vec![0.0; SIZE], vec![0.0; SIZE]);

        let (_success, _residual, _iters) = minres(
            &mut x,
            &Arows,
            &Acols,
            &Avals,
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
        );

        spmv(&mut error, &Arows, &Acols, &Avals, &x);
        error.scale_add(-1.0, 1.0, &b);

        let true_residual = dot(&error, &error).sqrt();

        error.linear_comb(1.0, &x, -1.0, &sol);
        let solution_error = dot(&error, &error).sqrt();

        assert!(true_residual <= TOLERANCE);
        assert!(solution_error <= REDUCED_TOL);
    }

    #[rstest]
    fn minres_2by2() {
        let maxiters: usize = 100;
        let mut plugin = StopWatchAndPrinter::new();
        const SIZE: usize = 2;

        let Arows: [usize; 3] = [0, 1, 0];
        let Acols: [usize; 3] = [0, 0, 1];
        let Avals: [f64; 3] = [1.0, 1.0, 1.0];

        let sol: [f64; SIZE] = [1.0, 1.0];

        let mut b: [f64; SIZE] = [0.0; SIZE];
        spmv(&mut b, &Arows, &Acols, &Avals, &sol);

        let (mut x, mut error) = (vec![0.0; SIZE], vec![0.0; SIZE]);

        let (_success, _residual, _iters) = minres(
            &mut x,
            &Arows,
            &Acols,
            &Avals,
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
        );

        spmv(&mut error, &Arows, &Acols, &Avals, &x);
        error.scale_add(-1.0, 1.0, &b);

        let true_residual = dot(&error, &error).sqrt();

        error.linear_comb(1.0, &x, -1.0, &sol);
        let solution_error = dot(&error, &error).sqrt();

        assert!(true_residual <= TOLERANCE);
        assert!(solution_error <= REDUCED_TOL);
    }

    #[rstest]
    fn minres_4by4() {
        const SIZE: usize = 4;
        let maxiters: usize = 2 * SIZE;
        let mut plugin = StopWatchAndPrinter::new();

        let Arows: [usize; 6] = [0, 2, 1, 3, 0, 1];
        let Acols: [usize; 6] = [0, 0, 1, 1, 2, 3];
        let Avals: [f64; 6] = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let sol: [f64; SIZE] = [1.0; SIZE];

        let mut b: [f64; SIZE] = [0.0; SIZE];
        spmv(&mut b, &Arows, &Acols, &Avals, &sol);

        let (mut x, mut error) = (vec![0.0; SIZE], vec![0.0; SIZE]);

        let (_success, _residual, _iters) = minres(
            &mut x,
            &Arows,
            &Acols,
            &Avals,
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
        );

        spmv(&mut error, &Arows, &Acols, &Avals, &x);
        error.scale_add(-1.0, 1.0, &b);

        let true_residual = dot(&error, &error).sqrt();

        error.linear_comb(1.0, &x, -1.0, &sol);
        let solution_error = dot(&error, &error).sqrt();

        assert!(true_residual <= TOLERANCE);
        assert!(solution_error <= REDUCED_TOL);
    }

    #[rstest]
    #[case("./data/nemeth26.csv")]
    #[case("./data/GHS_indef_qpband.csv")]
    #[case("./data/FIDAP_ex4.csv")]
    fn minres_mm(#[case] file_path: &str) {
        let (Arows, Acols, Avals, size, _nonzeros) = mm_read(file_path);

        let mut plugin = StopWatchAndPrinter::new();
        let mut x = vec![0.0; size];
        let mut sol: Vec<f64> = vec![1.0; size];
        let sol_norm: f64 = dot(&sol, &sol).sqrt();
        sol.scale(1.0 / sol_norm);

        let mut error = vec![0.0; size];
        let mut b: Vec<f64> = vec![0.0; size];
        spmv(&mut b, &Arows, &Acols, &Avals, &sol);

        for shift in [10_f64, 0_f64] {
            minres(
                &mut x,
                &Arows,
                &Acols,
                &Avals,
                &b,
                &shift,
                &TOLERANCE,
                &5_000,
                &mut plugin,
                &mut vec![0.0; size],
                &mut vec![0.0; size],
                &mut vec![0.0; size],
                &mut vec![0.0; size],
                &mut vec![0.0; size],
            );
        }

        spmv(&mut error, &Arows, &Acols, &Avals, &x);
        error.scale_add(-1.0, 1.0, &b);

        let true_residual = dot(&error, &error).sqrt();

        error.linear_comb(1.0, &x, -1.0, &sol);
        let solution_error = dot(&error, &error).sqrt();

        assert!(true_residual <= TOLERANCE);
        assert!(solution_error <= REDUCED_TOL);
    }
}
