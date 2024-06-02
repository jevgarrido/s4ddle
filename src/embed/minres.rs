#![allow(non_snake_case)]

use crate::embed::operators::*;
use libm::sqrt;

pub fn minres(
    x: &mut [f64],   // Solution vector (initial guess)
    mrows: &[usize], // Left-hand-side matrix: row indices
    mcols: &[usize], // Left-hand-side matrix: column indices
    mvals: &[f64],   // Left-hand-side matrix: values
    b: &[f64],       // Right-hand-side of the system
    tolerance: &f64,
    maxiters: &usize,
    aux_vector: &mut [f64],
    v: &mut [f64],
    v_next: &mut [f64],
    p_prev: &mut [f64],
    p: &mut [f64],
) {
    // v1 = r0 = b - A * x0
    v.linear_comb(1.0, b, -1.0, spmv(aux_vector, mrows, mcols, mvals, x));

    let norm_r0 = sqrt(dot(v, v));

    // Early return if initial guess meets tolerance
    if norm_r0 <= *tolerance {
        return;
    }

    // Make v unit length
    v.scale(1.0 / norm_r0);

    v_next.reset();
    p_prev.reset();
    p.reset();

    let mut ta: f64;
    let mut tb: f64;
    let mut tb_prev: f64 = 0.0;

    let mut X: f64;
    let mut Y: f64;

    let mut ua: f64;
    let mut ub: f64 = 0.0;
    let mut uc: f64 = 0.0;
    let mut uc_next: f64 = 0.0;

    let mut sin_product: f64 = 1.0;
    let mut alpha: f64;
    let mut residual: f64;

    let mut cos: f64;
    let mut cos_prev: f64 = 0.0;
    let mut cos_prev_prev: f64 = 0.0;
    let mut sin: f64;
    let mut sin_prev: f64 = 0.0;

    for kk in 1..=*maxiters {
        // Lanczos Iteration ---------------------
        spmv(aux_vector, mrows, mcols, mvals, v);
        ta = dot(v, aux_vector);

        v_next.scale_add(-tb_prev, -ta, v); // use tb
        v_next.add(1.0, aux_vector);

        tb = sqrt(dot(v_next, v_next)); // update tb after use
        v_next.scale(1.0 / tb);
        // ---------------------------------------
        //
        // Givens Rotation -----------------------
        if kk == 1 {
            X = ta;
        } else {
            X = tb_prev * sin_prev + ta * cos_prev;
        }

        ua = sqrt(X.powi(2) + tb.powi(2));

        cos = X / ua;
        sin = -tb / ua;

        if kk >= 2 {
            if kk == 2 {
                Y = tb_prev;
            } else {
                Y = tb_prev * cos_prev_prev;
            }

            ub = Y * cos_prev - ta * sin_prev;
            uc_next = -tb * sin_prev;
        }
        // ---------------------------------------
        //
        // Search direction and step size --------
        p.scale_add(-uc / ua, -ub / ua, p_prev);
        p.add(1.0 / ua, v);

        alpha = norm_r0 * sin_product * cos;
        residual = norm_r0 * sin_product * sin;

        x.add(alpha, p);

        if residual <= *tolerance {
            break;
        }
        // ---------------------------------------
        //
        // Close the loop ------------------------
        swap(v, v_next, aux_vector);
        tb_prev = tb;
        uc = uc_next;

        sin_product *= sin;
        (cos_prev_prev, cos_prev, sin_prev) = (cos_prev, cos, sin);

        swap(p_prev, p, aux_vector);
        // ---------------------------------------
    }
}

pub fn minres_precond() {}
