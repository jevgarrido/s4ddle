#![allow(non_snake_case)]

use crate::operators::*;
use crate::plugins::*;

/// Implementation of the preconditioned MINRES algorithm.
/// Coefficient matrix and preconditioner are NOT explicittly required.
/// Matrix-vector products are outsourced to you through a "black box" interface.
pub fn minres_black_box_precond<'a, T: Float>(
    mut x: &'a mut [T],                  // Solution vector (initial guess)
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

    // v1 = r0 = M * [ b - (A + shift * I ) * x0 ]
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

    p.reset();
    core::mem::swap(&mut x, &mut p);

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

            p.reset();
            core::mem::swap(&mut x, &mut p);

            x.reset();
            black_box.apply_precond_transpose(x, p);
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
