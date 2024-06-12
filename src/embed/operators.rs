#![allow(non_snake_case)]

/// Computes the sparse matrix vector (SpMV) multiplication for f64.
/// Matrix should be in COO format.
pub fn spmv(product: &mut [f64], mrows: &[usize], mcols: &[usize], mvals: &[f64], vector: &[f64]) {
    product.reset();

    for (idx, mval) in mvals.iter().enumerate() {
        product[mrows[idx]] += mval * vector[mcols[idx]];
    }
}

/// Computes the dot product between two vectors of type f64.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut product: f64 = 0.0;
    for ii in 0..x.len() {
        product += x[ii] * y[ii];
    }

    product
}

pub trait VectorManipulation {
    fn reset(&mut self);
    fn scale(&mut self, beta: f64) -> &mut Self;
    fn add(&mut self, beta: f64, y: &[f64]) -> &mut Self;
    fn scale_add(&mut self, alpha: f64, beta: f64, y: &[f64]) -> &mut Self;
    fn linear_comb(&mut self, a: f64, x: &[f64], b: f64, y: &[f64]) -> &mut Self;
}

impl VectorManipulation for [f64] {
    fn reset(&mut self) {
        for p in self.iter_mut() {
            *p = 0.0;
        }
    }
    fn scale(&mut self, beta: f64) -> &mut Self {
        for elm in self.iter_mut() {
            *elm *= beta;
        }
        self
    }
    fn add(&mut self, beta: f64, y: &[f64]) -> &mut Self {
        for (kk, elm) in self.iter_mut().enumerate() {
            *elm += beta * y[kk];
        }
        self
    }
    fn scale_add(&mut self, alpha: f64, beta: f64, y: &[f64]) -> &mut Self {
        for (kk, elm) in self.iter_mut().enumerate() {
            *elm = alpha * *elm + beta * y[kk];
        }
        self
    }
    fn linear_comb(&mut self, alpha: f64, x: &[f64], beta: f64, y: &[f64]) -> &mut Self {
        for (kk, elm) in self.iter_mut().enumerate() {
            *elm = alpha * x[kk] + beta * y[kk];
        }
        self
    }
}

pub trait Plugin {
    fn start(&mut self) {}
    fn peek(&mut self, _iters: &usize, _x: &[f64], _p: &[f64], _alpha: &f64, _residual: &f64) {}
    fn end(&mut self) {}
}

#[derive(Default)]
pub struct DoNothing;
impl Plugin for DoNothing {}

pub use fancy_plugin::StopWatchAndPrinter;

#[cfg(feature = "std")]
mod fancy_plugin {
    use crate::embed::operators::dot;

    use super::Plugin;
    use std::time::{Duration, Instant};

    pub struct StopWatchAndPrinter {
        now: Instant,           // Object of type "Instant"
        elapsed_time: Duration, // Object of type "Duration"
    }

    impl Default for StopWatchAndPrinter {
        fn default() -> Self {
            Self {
                now: Instant::now(),
                elapsed_time: Duration::new(0, 0),
            }
        }
    }

    impl Plugin for StopWatchAndPrinter {
        fn start(&mut self) {
            self.now = Instant::now();
            println!("");
            println!(
                "{:>5}  {:^13}  {:^13}  {:^13}",
                "iter", "residual", "step", "alpha"
            );
        }
        fn peek(&mut self, iter: &usize, _x: &[f64], p: &[f64], alpha: &f64, residual: &f64) {
            println!(
                "{iter:>5}  {residual:<13.7e}  {:<13.7e}  {:<13.7e}",
                dot(p, p).sqrt() * alpha.abs(),
                alpha.abs()
            );
        }
        fn end(&mut self) {
            self.elapsed_time = self.now.elapsed();
            println!("");
            println!("Elapsed time: {:.3e} secs", self.elapsed_time.as_secs_f64());
        }
    }
}
