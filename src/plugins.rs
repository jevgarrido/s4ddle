use crate::operators::*;

// ------------------------------------------------------------------------------------------------
// Black Box APIs
// ------------------------------------------------------------------------------------------------

/// API for hiden matrix-vector product kernel.
pub trait BlackBox<T: Float> {
    #[allow(non_snake_case)]
    fn apply_A(&self, product: &mut [T], vector: &[T]);
}

/// API for hiden matrix-vector products in algorithms that employ preconditioning of the form
/// $M A x = M b$ with $x = M^T y$, or, equivalently,
/// $M A M^T y = M b$ with $x = M^T y$.
pub trait BlackBoxPrecond<T: Float>: BlackBox<T> {
    fn apply_precond(&self, product: &mut [T], vector: &[T]);
    fn apply_precond_transpose(&self, product: &mut [T], vector: &[T]);
    fn apply_precond_inverse_transpose(&self, product: &mut [T], vector: &[T]);
}

// ------------------------------------------------------------------------------------------------
// Plugins for read-only interactivity with the algorithms
// ------------------------------------------------------------------------------------------------

/// Plugin for interactivity.
/// Implement this trait to enable custom interactive behaviour.
pub trait Plugin<T: Float> {
    fn start(&mut self);
    fn peek(&mut self, iters: &usize, x: &[T], p: &[T], alpha: &T, residual: &T);
    fn end(&mut self);
}

// ----------------------

/// A plugin implementation that does nothing.
/// This is the base struct for the decorator pattern employed for plugins.
/// Usage:
/// ```rust
/// # use s4ddle::plugins::DoNothing;
/// let mut plugin = DoNothing::new();
/// ```
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
    fn peek(&mut self, _iter: &usize, _x: &[T], _p: &[T], _alpha: &T, _residual: &T) {}
    #[inline(always)]
    fn end(&mut self) {}
}

/// Template struct that implements the decorator pattern for plugins.
///
/// Usage:
/// ```rust
/// # use s4ddle::plugins::*;
/// # use s4ddle::operators::*;
/// let mut base_plugin = DoNothing::new();
/// let mut decorated_plugin: DoNothingDecorator<'_,f64,_> =
///     DoNothingDecorator::extend(&mut base_plugin);
/// ```
pub struct DoNothingDecorator<'a, T: Float, P: Plugin<T>> {
    plugin: &'a mut P,
    phantom: core::marker::PhantomData<&'a T>,
}

impl<'a, T: Float, P: Plugin<T>> DoNothingDecorator<'a, T, P> {
    pub fn extend(plugin: &'a mut P) -> Self {
        Self { plugin, phantom: core::marker::PhantomData }
    }
}

impl<T: Float, P: Plugin<T>> Plugin<T> for DoNothingDecorator<'_, T, P> {
    #[inline(always)]
    fn start(&mut self) {
        self.plugin.start();
    }
    #[inline(always)]
    fn peek(&mut self, iter: &usize, x: &[T], p: &[T], alpha: &T, residual: &T) {
        self.plugin.peek(iter, x, p, alpha, residual);
    }
    #[inline(always)]
    fn end(&mut self) {
        self.plugin.end();
    }
}

// --------------------------------------------------------------------------------------------

/// A plugin implementation that collects residuals using the decorator pattern.
/// This plugin collects the (2-Norm) value of || ***b*** - **A** ***x*** || at each iteration.
///
/// This struct does not own the data on which it operates, so the values are stored in an external mutable vector.
///
/// Example usage:
/// ```rust
/// # use s4ddle::plugins::*;
/// # use s4ddle::operators::*;
/// const NUM_ITERATIONS: usize = 32;
/// let mut vector_of_residuals = [0.0_f64;NUM_ITERATIONS];
/// let mut base_plugin = DoNothing::new();
/// let mut residual_collector_plugin: CollectResiduals<'_,f64,_> =
///     CollectResiduals::extend(&mut base_plugin, &mut vector_of_residuals);
///
/// println!("{:?}", vector_of_residuals);
/// ```
pub struct CollectResiduals<'a, T: Float, P: Plugin<T>> {
    plugin: &'a mut P,
    //
    residuals: &'a mut [T],
}

impl<'a, T: Float, P: Plugin<T>> CollectResiduals<'a, T, P> {
    pub fn extend(plugin: &'a mut P, residuals: &'a mut [T]) -> Self {
        Self { plugin, residuals }
    }
}

impl<T: Float, P: Plugin<T>> Plugin<T> for CollectResiduals<'_, T, P> {
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

/// A plugin implementation that collects solution errors using the decorator pattern.
/// This plugin collects the value of || ***x*** - ***<ins>x</ins>*** || (2-Norm) at each
/// iteration, where ***<ins>x</ins>*** denotes the (known) solution of the system.
///
/// This struct does not own the data on which it operates, so the values are stored in external mutable vectors.
///
/// Example Usage:
/// ```rust
/// # use s4ddle::plugins::*;
/// # use s4ddle::operators::*;
/// const NUM_ITERATIONS: usize = 32;
/// const PROB_SIZE: usize = 20;
///
/// let mut vector_of_errors = [0.0_f64;NUM_ITERATIONS];
/// let mut vector_of_differences = [0.0_f64;PROB_SIZE];
/// let solution = [1.0_f64;PROB_SIZE];
///
/// let mut base_plugin = DoNothing::new(); // <- Base decorator plugin
///
/// let mut residual_collector_plugin: CollectSolutionErrors<'_,f64,_> =
///     CollectSolutionErrors::extend(
///         &mut base_plugin,
///         &solution,
///         &mut vector_of_errors,
///         &mut vector_of_differences);
///
/// println!("{:?}", vector_of_errors);
/// ```
///
pub struct CollectSolutionErrors<'a, T: Float, P: Plugin<T>> {
    plugin: &'a mut P,
    //
    solution: &'a [T],
    errors: &'a mut [T],
    diff: &'a mut [T],
}

impl<'a, T: Float, P: Plugin<T>> CollectSolutionErrors<'a, T, P> {
    pub fn extend(plugin: &'a mut P, solution: &'a [T], errors: &'a mut [T], diff: &'a mut [T]) -> Self {
        Self { plugin, solution, errors, diff }
    }
}

impl<T: Float, P: Plugin<T>> Plugin<T> for CollectSolutionErrors<'_, T, P> {
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

#[cfg(feature = "std")]
pub use std_plugins::*;

#[cfg(feature = "std")]
mod std_plugins {
    use super::*;

    /// A plugin implementation that measures execution time.
    /// This Plugin uses Instant and Duration from the standard library
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

    /// A plugin implementation that prints out some information at every iteration.
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
