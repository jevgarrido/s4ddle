// MINRES
//
pub fn minres() {}

// SYMMLQ
//
pub fn symmlq() {}

/// Conjugate residual method, **CR**
/// Reference:
///     Luenberger, David G., "The Conjugate Residual Method for Constrained Minimization Problems", SIAM Journal on Numerical Analysis, Vol. 7, No. 3, September 1970, doi:[10.1137/0707032](https://www.doi.org/10.1137/0707032)
pub fn cr() {}

#[cfg(test)]
mod tests {
    #[test]
    fn test_minres() {
        let g = vec![1.0, 1.0];
        println!("{:?}", g);
    }
}
