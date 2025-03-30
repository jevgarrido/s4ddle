#![allow(non_snake_case)]

use crate::operators::*;

pub fn ruiz_scaling<T: Float>(
    precond: &mut [T],
    _Arows: &[usize],
    Acols: &[usize],
    Avals: &[T],
    max_iters: &usize,
    tol: &T,
    row_norms: &mut [T],
) {
    for elm in precond.iter_mut() {
        *elm = T::one();
    }
    let mut convergence_achieved = true;

    for _ in 0..*max_iters {
        row_norms.reset();

        for (col, val) in Acols.iter().zip(Avals.iter()) {
            row_norms[*col] = row_norms[*col] + (*val * precond[*col]).powi(2);
        }

        for (idx, elm) in row_norms.iter_mut().enumerate() {
            *elm = precond[idx].abs() * elm.sqrt();
        }

        for elm in row_norms.iter() {
            if (*elm - T::one()).abs() >= *tol {
                convergence_achieved = false;
                break;
            }
        }

        if convergence_achieved {
            break;
        } else {
            convergence_achieved = true;
        }

        for (idx, elm) in precond.iter_mut().enumerate() {
            *elm = T::one() / row_norms[idx].sqrt();
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use rstest::*;
    use crate::readwrite::*;

    #[rstest]
    #[case("./data/nemeth01.csv")] // Condition # : 1.399218e+02
    #[case("./data/nemeth26.csv")] // Condition # : 1.000004e+00
    #[case("./data/GHS_indef_qpband.csv")] // Condition # : 6.436577e+00
    #[case("./data/GHS_indef_tuma2.csv")] // Condition # : 1.701266e+03
    #[case("./data/GHS_indef_linverse.csv")] // Condition # : 3.946608e+03
    #[case("./data/FIDAP_ex4.csv")] // Condition # : 2.386583e+03
    fn test_matrix_market(#[case] file_path: &str) {
        let (Arows, Acols, Avals, size, _nonzeros) = mm_read(file_path);

        let max_iters = 12;
        let tol = 1e-12;
        let mut precond_diag = vec![0.0; size];

        ruiz_scaling(&mut precond_diag, &Arows, &Acols, &Avals, &max_iters, &tol, &mut vec![0.0; size]);
    }
}
