use crate::embed::operators::*;

pub fn cr(
    x: &mut [f64],
    mrows: &[usize],
    mcols: &[usize],
    mvals: &[f64],
    b: &[f64],
    tol: &f64,
    maxiters: &usize,
    r: &mut [f64],
    mut pk: &mut [f64],
    mut pkm1: &mut [f64],
    mut sk: &mut [f64],
    mut skm1: &mut [f64],
    mut aux: &mut [f64],
) -> (bool, usize, f64) {
    let squared_tol = tol * tol;
    let mut alpha: f64;
    let mut beta: f64;

    pkm1.reset();
    aux.reset();

    // r = b - A * x;
    r.linear_comb(1.0, b, -1.0, spmv(pk, mrows, mcols, mvals, x));

    // p_k = r;
    pk.copy_from_slice(&r);

    // s_k = A * p_k;
    spmv(sk, mrows, mcols, mvals, pk);

    for _ii in 1..=*maxiters {
        alpha = dot(r, sk) / dot(sk, sk);

        x.add(alpha, pk);
        r.add(-alpha, sk);

        if dot(r, r) <= squared_tol {
            break;
        }

        pkm1.copy_from_slice(pk);
        skm1.copy_from_slice(sk);

        if alpha * alpha > squared_tol {
            // alpha != 0

            // aux = A * r;
            spmv(aux, mrows, mcols, mvals, r);

            beta = dot(aux, sk) / dot(sk, sk);

            pk.linear_comb(1.0, r, -beta, pkm1);
            sk.linear_comb(1.0, aux, -beta, skm1);
        } else {
            // alpha == 0

            // aux = A * ( A * r ) = A * ( A*pkm1 ) = A * skm1;
            spmv(aux, mrows, mcols, mvals, skm1);

            // gama = dot( aux, ) / dot();
            // delta = dot() / dot();
        }
    }

    (true, 1, dot(r, r))
}

pub fn cr_precond() {
    todo!()
}
