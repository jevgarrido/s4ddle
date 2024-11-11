#![allow(non_snake_case)]

use core::ops::{AddAssign, SubAssign, MulAssign, DivAssign, RemAssign};
use core::iter::Sum;
use core::fmt::{LowerExp, UpperExp, Debug};
use core::marker::Copy;
use num_traits::float::Float as NumFloat;

/// Trait for generic declarations of floating point types (f32 and f64).
pub trait Float:
    NumFloat + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Sum + LowerExp + UpperExp + Debug + Copy
{
}
impl Float for f64 {}
impl Float for f32 {}

pub trait VectorManipulation<T: Float> {
    fn reset(&mut self);
    fn scale(&mut self, beta: T) -> &mut Self;
    fn add(&mut self, beta: T, y: &[T]) -> &mut Self;
    fn scale_add(&mut self, alpha: T, beta: T, y: &[T]) -> &mut Self;
    fn linear_comb(&mut self, a: T, x: &[T], b: T, y: &[T]) -> &mut Self;
    fn norm_1(&self) -> T;
    fn norm_2(&self) -> T;
    fn norm_inf(&self) -> T;
}

impl<T: Float> VectorManipulation<T> for [T] {
    fn reset(&mut self) {
        for p in self.iter_mut() {
            *p = T::zero();
        }
    }
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
    fn scale_add(&mut self, alpha: T, beta: T, y: &[T]) -> &mut Self {
        for (kk, elm) in self.iter_mut().enumerate() {
            *elm *= alpha;
            *elm += beta * y[kk];
        }
        self
    }
    fn linear_comb(&mut self, alpha: T, x: &[T], beta: T, y: &[T]) -> &mut Self {
        for (kk, elm) in self.iter_mut().enumerate() {
            *elm = alpha * x[kk] + beta * y[kk];
        }
        self
    }

    #[inline(always)]
    fn norm_1(&self) -> T {
        self.iter().map(|x| x.abs()).sum::<T>()
    }

    #[inline(always)]
    fn norm_2(&self) -> T {
        self.iter().map(|x| x.powi(2)).sum::<T>().sqrt()
    }

    #[inline(always)]
    fn norm_inf(&self) -> T {
        self.iter().max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap().abs()
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

// ------------------------------------------------------------------------------------------------
// Plugins
// ------------------------------------------------------------------------------------------------

/// API for "hiden" matrix-vector products.
/// Default implementation for an identity preconditioner.
pub trait BlackBox<T: Float> {
    fn apply_A(&self, product: &mut [T], vector: &[T]);

    #[inline(always)]
    fn apply_precond(&self, product: &mut [T], vector: &[T]) {
        product.copy_from_slice(vector);
    }

    #[inline(always)]
    fn apply_precond_transpose(&self, product: &mut [T], vector: &[T]) {
        product.copy_from_slice(vector);
    }

    #[inline(always)]
    fn apply_precond_inverse_transpose(&self, product: &mut [T], vector: &[T]) {
        product.copy_from_slice(vector);
    }
}

/// Plugin for interactivity.
/// Implement this trait if you wish to enable custom interactive behaviour.
pub trait Plugin<T: Float> {
    fn start(&mut self);
    fn peek(&mut self, iters: &usize, x: &[T], p: &[T], alpha: &T, residual: &T);
    fn end(&mut self);
}

/// Place-holder that implements the Plugin trait.
pub struct DoNothing;

impl DoNothing {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T: Float> Plugin<T> for DoNothing {
    #[inline(always)]
    fn start(&mut self) {}
    #[inline(always)]
    fn peek(&mut self, _iters: &usize, _x: &[T], _p: &[T], _alpha: &T, _residual: &T) {}
    #[inline(always)]
    fn end(&mut self) {}
}

#[cfg(feature = "std")]
pub struct StopWatchAndPrinter {
    now: std::time::Instant,
    elapsed_time: std::time::Duration,
}

#[cfg(feature = "std")]
impl StopWatchAndPrinter {
    pub fn new() -> Self {
        Self { now: std::time::Instant::now(), elapsed_time: std::time::Duration::new(0, 0) }
    }
}

#[cfg(feature = "std")]
impl<T: Float> Plugin<T> for StopWatchAndPrinter {
    fn start(&mut self) {
        self.now = std::time::Instant::now();
        println!("");
        println!("{:>5}  {:^13}  {:^13}  {:^13}", "iter", "residual", "step", "alpha");
    }
    #[inline(always)]
    fn peek(&mut self, iter: &usize, _x: &[T], p: &[T], alpha: &T, residual: &T) {
        println!("{iter:>5}  {residual:<13.7e}  {:<13.7e}  {:<13.7e}", dot(p, p).sqrt() * alpha.abs(), alpha.abs());
    }
    fn end(&mut self) {
        self.elapsed_time = self.now.elapsed();
        println!("");
        println!("Elapsed time: {:.3e} secs", self.elapsed_time.as_secs_f64());
    }
}
