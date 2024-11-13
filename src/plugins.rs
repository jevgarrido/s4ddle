#![allow(non_snake_case)]

use crate::operators::*;

// ------------------------------------------------------------------------------------------------
// Black Box APIs
// ------------------------------------------------------------------------------------------------

/// API for hiden matrix-vector product kernel.
pub trait BlackBox<T: Float> {
    fn apply_A(&self, product: &mut [T], vector: &[T]);
}

/// API for hiden matrix-vector products in algorithms that employ preconditioning of the form.
/// $M A x = M b$ with $x = M^T y$, or, equivalently,
/// $M A M^T y = M b$ with $x = M^T y$.
/// Default implementation for the identity preconditioner.
pub trait BlackBoxPrecond<T: Float>: BlackBox<T> {
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

// ------------------------------------------------------------------------------------------------
// APIS for interactivity with the algorithms (read-only)
// ------------------------------------------------------------------------------------------------

/// Plugin for interactivity.
/// Implement this trait if you wish to enable custom interactive behaviour.
pub trait Plugin<T: Float> {
    fn start(&mut self);
    fn peek(&mut self, iters: &usize, x: &[T], p: &[T], alpha: &T, residual: &T);
    fn end(&mut self);
}

/// A plugin implementation that does nothing.
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

/// A plugin implementation that measures execution time and prints out some information at every
/// iteration.
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
    }
    #[inline(always)]
    fn peek(&mut self, iter: &usize, x: &[T], p: &[T], alpha: &T, residual: &T) {
        if iter % 10 == 0 || *iter == 1usize {
            println!("");
            println!("{:>5}  {:^13}  {:^13}  {:^13}  {:^13}", "iter", "residual", "step", "alpha", "x[0]");
        }
        println!(
            "{iter:>5}  {residual:<13.7e}  {:<13.7e}  {:<13.7e}  {:<13.7e}",
            p.norm_2() * alpha.abs(),
            alpha.abs(),
            x[0]
        );
    }
    fn end(&mut self) {
        self.elapsed_time = self.now.elapsed();
        println!("");
        println!("Elapsed time: {:.3e} secs", self.elapsed_time.as_secs_f64());
    }
}

/// A plugin implementation that collects residuals and writes them to disk.
#[cfg(feature = "std")]
pub struct CollectResiduals<'a, T: Float> {
    path: String,
    residuals: &'a mut [T],
    total_iters: usize,
}

#[cfg(feature = "std")]
impl<'a, T: Float> CollectResiduals<'a, T> {
    pub fn new(vec: &'a mut [T], path: &str) -> Self {
        Self { path: path.to_owned(), residuals: vec, total_iters: 0usize }
    }
}

#[cfg(feature = "std")]
impl<T: Float> Plugin<T> for CollectResiduals<'_, T> {
    #[inline(always)]
    fn start(&mut self) {}
    #[inline(always)]
    fn peek(&mut self, iter: &usize, _x: &[T], _p: &[T], _alpha: &T, residual: &T) {
        self.total_iters = *iter;
        self.residuals[*iter] = *residual;
    }
    #[inline(always)]
    fn end(&mut self) {
        use std::io::Write;

        let f = std::fs::File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(self.path.as_str())
            .ok()
            .unwrap();
        let mut f = std::io::BufWriter::new(f);

        for &val in self.residuals[1..=self.total_iters].iter() {
            writeln!(&mut f, "{val}").ok();
        }
    }
}
