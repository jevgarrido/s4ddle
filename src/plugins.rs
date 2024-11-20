use crate::operators::*;

// ------------------------------------------------------------------------------------------------
// Black Box APIs
// ------------------------------------------------------------------------------------------------

/// API for hiden matrix-vector product kernel.
pub trait BlackBox<T: Float> {
    #[allow(non_snake_case)]
    fn apply_A(&self, product: &mut [T], vector: &[T]);
}

/// API for hiden matrix-vector products in algorithms that employ preconditioning of the form.
/// $M A x = M b$ with $x = M^T y$, or, equivalently,
/// $M A M^T y = M b$ with $x = M^T y$.
/// Default implementation for the identity preconditioner.
pub trait BlackBoxPrecond<T: Float>: BlackBox<T> {
    fn apply_precond(&self, product: &mut [T], vector: &[T]);

    fn apply_precond_transpose(&self, product: &mut [T], vector: &[T]);

    fn apply_precond_inverse_transpose(&self, product: &mut [T], vector: &[T]);
}

// ------------------------------------------------------------------------------------------------
// Plugins for read-only interactivity with the algorithms
// ------------------------------------------------------------------------------------------------

/// Plugin for interactivity.
/// Implement this trait if you wish to enable custom interactive behaviour.
pub trait Plugin<T: Float> {
    fn start(&mut self);
    fn peek(&mut self, iters: &usize, x: &[T], p: &[T], alpha: &T, residual: &T);
    fn end(&mut self);
}

// ----------------------

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

// ------------------------------------------------------------------------------------------------
// Plugins that implement the decorator pattern
// ------------------------------------------------------------------------------------------------

#[cfg(feature = "std")]
pub use std_plugins::*;

#[cfg(feature = "std")]
mod std_plugins {
    use super::*;

    /// A plugin implementation that collects residuals.
    /// This implementation employs the decorator pattern.
    pub struct CollectResiduals<'a, T: Float> {
        plugin: &'a mut dyn Plugin<T>,
        //
        residuals: &'a mut [T],
    }

    impl<'a, T: Float> CollectResiduals<'a, T> {
        pub fn extend(plugin: &'a mut dyn Plugin<T>, residuals: &'a mut [T]) -> Self {
            Self { plugin, residuals }
        }
    }

    impl<T: Float> Plugin<T> for CollectResiduals<'_, T> {
        #[inline(always)]
        fn start(&mut self) {
            self.plugin.start();
        }
        #[inline(always)]
        fn peek(&mut self, iter: &usize, x: &[T], p: &[T], alpha: &T, residual: &T) {
            self.residuals[*iter] = *residual;

            self.plugin.peek(iter, x, p, alpha, residual);
        }
        #[inline(always)]
        fn end(&mut self) {
            self.plugin.end();
        }
    }

    // --------------------------------------------------------------------------------------------

    /// A plugin implementation that collects solution errors.
    /// This implementation employs the decorator pattern.
    pub struct CollectSolutionErrors<'a, T: Float> {
        plugin: &'a mut dyn Plugin<T>,
        //
        solution: &'a [T],
        errors: &'a mut [T],
        diff: &'a mut [T],
    }

    impl<'a, T: Float> CollectSolutionErrors<'a, T> {
        pub fn extend(
            plugin: &'a mut dyn Plugin<T>,
            solution: &'a [T],
            errors: &'a mut [T],
            diff: &'a mut [T],
        ) -> Self {
            Self { plugin, solution, errors, diff }
        }
    }

    impl<T: Float> Plugin<T> for CollectSolutionErrors<'_, T> {
        #[inline(always)]
        fn start(&mut self) {
            self.plugin.start();
        }
        #[inline(always)]
        fn peek(&mut self, iter: &usize, x: &[T], p: &[T], alpha: &T, residual: &T) {
            self.diff.reset();
            self.diff.copy_from_slice(x);
            self.diff.add(-T::one(), self.solution);

            self.errors[*iter] = self.diff.norm_2();

            self.plugin.peek(iter, x, p, alpha, residual);
        }
        #[inline(always)]
        fn end(&mut self) {
            self.plugin.end();
        }
    }

    // --------------------------------------------------------------------------------------------

    /// A plugin implementation that measures execution time.
    /// This Plugin uses Instant and Duration from the standard library
    /// This implementation employs the decorator pattern.
    pub struct StopWatch<'a, T: Float> {
        plugin: &'a mut dyn Plugin<T>,
        //
        now: std::time::Instant,
        elapsed_time: std::time::Duration,
    }

    impl<'a, T: Float> StopWatch<'a, T> {
        pub fn extend(plugin: &'a mut dyn Plugin<T>) -> Self {
            Self { plugin, now: std::time::Instant::now(), elapsed_time: std::time::Duration::new(0, 0) }
        }
        pub fn elapsed_time_in_seconds(&self) -> f64 {
            self.elapsed_time.as_secs_f64()
        }
        pub fn elapsed_time_in_minutes(&self) -> f64 {
            self.elapsed_time.as_secs_f64() / 60.0
        }
    }

    impl<T: Float> Plugin<T> for StopWatch<'_, T> {
        #[inline(always)]
        fn start(&mut self) {
            self.now = std::time::Instant::now();

            self.plugin.start();
        }
        #[inline(always)]
        fn peek(&mut self, iter: &usize, x: &[T], p: &[T], alpha: &T, residual: &T) {
            self.plugin.peek(iter, x, p, alpha, residual);
        }
        #[inline(always)]
        fn end(&mut self) {
            self.elapsed_time = self.now.elapsed();

            self.plugin.end();
        }
    }

    // --------------------------------------------------------------------------------------------

    /// A plugin implementation that measures execution time and prints out some information at every
    /// iteration.
    /// This Plugin uses Instant and Duration from the standard library
    pub struct ConsoleLogger<'a, T: Float> {
        plugin: &'a mut dyn Plugin<T>,
    }

    impl<'a, T: Float> ConsoleLogger<'a, T> {
        pub fn extend(plugin: &'a mut dyn Plugin<T>) -> Self {
            Self { plugin }
        }
    }

    impl<T: Float> Plugin<T> for ConsoleLogger<'_, T> {
        fn start(&mut self) {
            self.plugin.start();
        }
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

            self.plugin.peek(iter, x, p, alpha, residual);
        }
        fn end(&mut self) {
            self.plugin.end();
        }
    }
}
