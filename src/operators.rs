use core::ops::{AddAssign, SubAssign, MulAssign, DivAssign, RemAssign};
use core::iter::{Sum, Product};
use core::fmt::{LowerExp, UpperExp, Debug, Display};
use core::marker::Copy;
use core::clone::Clone;
use core::default::Default;
use num_traits::float::Float as NumFloat;

/// Trait for generic declarations of floating point types (f32 and f64).
pub trait Float:
    NumFloat
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + Sum
    + Product
    + LowerExp
    + UpperExp
    + Debug
    + Display
    + Copy
    + Clone
    + Default
{
}
impl Float for f64 {}
impl Float for f32 {}

pub trait FloatSliceOps<T: Float> {
    fn norm_1(&self) -> T;
    fn norm_2(&self) -> T;

    fn reset(&mut self) -> &mut Self;
    fn scale(&mut self, beta: T) -> &mut Self;

    fn add(&mut self, beta: T, y: &[T]) -> &mut Self;
    fn scale_add(&mut self, alpha: T, beta: T, y: &[T]) -> &mut Self;
    fn linear_comb(&mut self, a: T, x: &[T], b: T, y: &[T]) -> &mut Self;

    fn dot(&self, other: &[T]) -> T;
    fn spmv(&mut self, mrows: &[usize], mcols: &[usize], mvals: &[T], vector: &[T]) -> &mut Self;
}

impl<T: Float> FloatSliceOps<T> for [T] {
    #[inline(always)]
    fn norm_1(&self) -> T {
        self.iter().map(|x| x.abs()).sum::<T>()
    }

    #[inline(always)]
    fn norm_2(&self) -> T {
        self.iter().map(|x| x.powi(2)).sum::<T>().sqrt()
    }

    #[inline(always)]
    fn reset(&mut self) -> &mut Self {
        for elm in self.iter_mut() {
            *elm = T::zero();
        }
        self
    }

    #[inline(always)]
    fn scale(&mut self, beta: T) -> &mut Self {
        for elm in self.iter_mut() {
            *elm *= beta;
        }
        self
    }

    #[inline(always)]
    fn add(&mut self, beta: T, y: &[T]) -> &mut Self {
        for (kk, elm) in self.iter_mut().enumerate() {
            *elm += beta * y[kk];
        }
        self
    }

    #[inline(always)]
    fn scale_add(&mut self, alpha: T, beta: T, y: &[T]) -> &mut Self {
        for (kk, elm) in self.iter_mut().enumerate() {
            *elm *= alpha;
            *elm += beta * y[kk];
        }
        self
    }

    /// Linear combination of two vectors.
    /// The original contents of the mutable input vector are disregarded and overwritten.
    #[inline(always)]
    fn linear_comb(&mut self, alpha: T, x: &[T], beta: T, y: &[T]) -> &mut Self {
        for (kk, elm) in self.iter_mut().enumerate() {
            *elm = alpha * x[kk] + beta * y[kk];
        }
        self
    }

    /// Computes the dot product between two vectors.
    /// Vectors should have the same number of elements.
    #[inline(always)]
    fn dot(&self, other: &[T]) -> T {
        self.iter().zip(other.iter()).map(|(x, y)| *x * *y).sum::<T>()
    }

    /// Computes the sparse matrix-vector (SpMV) multiplication.
    /// Matrix should be in COO format.
    #[inline(always)]
    fn spmv(&mut self, mrows: &[usize], mcols: &[usize], mvals: &[T], vector: &[T]) -> &mut Self {
        for ((row, col), val) in mrows.iter().zip(mcols.iter()).zip(mvals.iter()) {
            self[*row] += *val * vector[*col];
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_dot_orthogonal_vectors() {
        let sz = 100;
        let mut v1: Vec<f64> = vec![0.0; sz];
        let mut v2: Vec<f64> = vec![0.0; sz];

        for (kk, val) in v1.iter_mut().enumerate() {
            if kk % 2 == 0 {
                *val = 1.0;
            }
        }

        v2.clone_from_slice(&v1);
        v2.rotate_right(1);

        // Check that vectors are non-zero
        assert!(v1.norm_2() > 0.0);
        assert!(v2.norm_2() > 0.0);

        // Check that the dot product is zero.
        assert_eq!(v1.dot(&v2), 0.0);
    }

    #[rstest]
    fn test_dot_sum_of_integers() {
        let sz: usize = 100;

        // Vector of ones
        let v1: Vec<f64> = vec![1.0; sz];

        // Vector of increasing integers values
        let mut v2: Vec<f64> = vec![0.0; sz];

        for (kk, val) in v2.iter_mut().enumerate() {
            *val = kk as f64 + 1.0;
        }

        // Dot product equals the sum of elements of v2
        assert_eq!(v1.dot(&v2), v2.iter().sum());
    }

    #[rstest]
    fn test_dot_norm() {
        let sz: usize = 100;
        let v1: Vec<f64> = vec![-2.0; sz];

        // Check that the dot product matches the 2-Norm squared.
        assert_eq!(v1.dot(&v1), v1.norm_2().powi(2));
    }

    #[rstest]
    fn test_spmv_identity() {
        let sz: usize = 100;
        let mut prod: Vec<f64> = vec![0.0; sz];

        // Construct the identity matrix
        let rows: Vec<usize> = (0..sz).collect();
        let cols: Vec<usize> = (0..sz).collect();
        let vals: Vec<f64> = vec![1.0; sz];

        let v: Vec<f64> = vec![-1.0; sz];

        prod.spmv(&rows, &cols, &vals, &v);

        // Check that the product equals the input vector.
        assert_eq!(prod, v);
    }

    #[rstest]
    fn test_spmv_anti_identity() {
        let sz: usize = 100;
        let mut prod: Vec<f64> = vec![0.0; sz];

        // Construct the identity matrix
        let rows: Vec<usize> = (0..sz).collect();
        let cols: Vec<usize> = (0..sz).rev().collect();
        let vals: Vec<f64> = vec![1.0; sz];

        let mut v: Vec<f64> = vec![0.0; sz];

        for (kk, val) in v.iter_mut().enumerate() {
            *val = -(kk as f64 + 1.0);
        }

        // Apply anti-identity matrix
        prod.spmv(&rows, &cols, &vals, &v);

        // Check that the product equals the reversed input vector.
        v.reverse();
        assert_eq!(prod, v);
    }

    #[rstest]
    fn test_dense_lower_triangular_matrix() {
        let sz: usize = 100;

        let mut prod: Vec<f64> = vec![0.0; sz];
        let v: Vec<f64> = vec![1.0; sz];

        // Construct dense lower triangular matrix of ones
        let mut rows: Vec<usize> = vec![0; sz * (sz + 1) / 2];
        let mut cols: Vec<usize> = vec![0; sz * (sz + 1) / 2];
        let vals: Vec<f64> = vec![1.0; sz * (sz + 1) / 2];

        let mut kk: usize = 0;

        for ii in 0..sz {
            for jj in 0..sz {
                if ii >= jj {
                    rows[kk] = ii;
                    cols[kk] = jj;

                    kk += 1;
                }
            }
        }

        // Apply lower-triangular matrix
        prod.spmv(&rows, &cols, &vals, &v);

        // Construct the expected result: increasing sequence of integers
        let mut expected_result: Vec<f64> = vec![0.0; sz];

        for (kk, val) in expected_result.iter_mut().enumerate() {
            *val = (kk + 1) as f64;
        }

        // Check that the product equals the expected result
        assert_eq!(prod, expected_result);
    }
}
