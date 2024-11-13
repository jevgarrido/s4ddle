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
    fn max(&self) -> T;

    fn reset(&mut self) -> &mut Self;
    fn scale(&mut self, beta: T) -> &mut Self;

    fn add(&mut self, beta: T, y: &[T]) -> &mut Self;
    fn scale_add(&mut self, alpha: T, beta: T, y: &[T]) -> &mut Self;
    fn linear_comb(&mut self, a: T, x: &[T], b: T, y: &[T]) -> &mut Self;
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
    fn max(&self) -> T {
        self.iter().max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap().abs()
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

    #[inline(always)]
    fn linear_comb(&mut self, alpha: T, x: &[T], beta: T, y: &[T]) -> &mut Self {
        for (kk, elm) in self.iter_mut().enumerate() {
            *elm = alpha * x[kk] + beta * y[kk];
        }
        self
    }
}

/// Computes the sparse matrix-vector (SpMV) multiplication.
/// $$y = A v$$
/// Matrix should be in COO format.
/// Can be called with input types f32 and f64.
#[inline(always)]
pub fn spmv<T: Float>(product: &mut [T], mrows: &[usize], mcols: &[usize], mvals: &[T], vector: &[T]) {
    product.reset();
    for ((row, col), val) in mrows.iter().zip(mcols.iter()).zip(mvals.iter()) {
        product[*row] += *val * vector[*col];
    }
}

/// Computes the sparse matrix-vector (SpMV) multiplication with a transposed matrix.
/// $$y = A^T v$$
/// Matrix should be in COO format.
/// Can be called with input types f32 and f64.
#[inline(always)]
pub fn spmv_transpose<T: Float>(product: &mut [T], mrows: &[usize], mcols: &[usize], mvals: &[T], vector: &[T]) {
    product.reset();
    for ((col, row), val) in mrows.iter().zip(mcols.iter()).zip(mvals.iter()) {
        product[*row] += *val * vector[*col];
    }
}

/// Computes the dot product between two vectors. Vectors should have the same number of elements.
/// Can be called with input types f32 and f64.
#[inline(always)]
pub fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum::<T>()
}
