use s4ddle::*;
use s4ddle::operators::*;
use s4ddle::plugins::*;

fn sierpinski_matrix(nrows: &usize, ncols: &usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    for r in 0..*nrows {
        for c in 0..*ncols {
            if (r & c) == 0 {
                rows.push(r);
                cols.push(c);
                vals.push(1.0);
                print!("* ");
            } else {
                print!("  ");
            }

            if c == (*ncols - 1) {
                print!("\n");
            }
        }
    }
    (rows, cols, vals)
}

struct QuadraticProgramBlackBox<'a, T: Float> {
    // constraint coefficient matrix
    cm_rows: &'a [usize],
    cm_cols: &'a [usize],
    cm_vals: &'a [T],

    // problem dimensions
    var_sz: usize,
    constraint_sz: usize,
}

impl<'a, T: Float> BlackBox<T> for QuadraticProgramBlackBox<'a, T> {
    fn apply_A(&self, product: &mut [T], vector: &[T]) {
        product.reset();
        // First part of product
        spmv(
            product,
            self.cm_cols,
            self.cm_rows,
            self.cm_vals,
            &vector[self.var_sz..(self.var_sz + self.constraint_sz)],
        );

        product[0..self.var_sz].add(T::one(), &vector[0..self.var_sz]);

        // Second part of product
        spmv(
            &mut product[self.var_sz..(self.var_sz + self.constraint_sz)],
            self.cm_rows,
            self.cm_cols,
            self.cm_vals,
            vector,
        );
    }
}

fn main() {
    // Minimize
    //      1/2 x^T x
    // subject to
    //      A x = 1
    // where A is a wide Sierpinski matrix

    let constraint_sz: usize = 2i32.pow(4) as usize;
    let var_sz: usize = 2 * constraint_sz;

    // Specify diagonal shift
    let shift: f64 = 0.0;

    // Specify tolerance
    let tol: f64 = 1e-12;

    // Specify max iterations allowed
    let max_iters = 10_000;

    // Select an appropriate plugin.
    // let mut plugin = DoNothing::new();
    let mut plugin = StopWatchAndPrinter::new();

    // System of equality constraints: coefficient matrix has a Sierpinski triangle pattern
    let (constraint_rows, constraint_cols, constraint_vals) = sierpinski_matrix(&constraint_sz, &var_sz);
    let constraint_rhs: Vec<f64> = vec![1.0; constraint_sz];

    let black_box = QuadraticProgramBlackBox {
        cm_rows: &constraint_rows,
        cm_cols: &constraint_cols,
        cm_vals: &constraint_vals,
        var_sz,
        constraint_sz,
    };

    let mut sol: Vec<f64> = vec![0.0; var_sz + constraint_sz];
    let mut b: Vec<f64> = vec![0.0; var_sz];

    b.append(&mut constraint_rhs.clone());

    let mut aux1: Vec<f64> = vec![0.0; var_sz + constraint_sz];
    let mut aux2: Vec<f64> = vec![0.0; var_sz + constraint_sz];
    let mut aux3: Vec<f64> = vec![0.0; var_sz + constraint_sz];
    let mut aux4: Vec<f64> = vec![0.0; var_sz + constraint_sz];
    let mut aux5: Vec<f64> = vec![0.0; var_sz + constraint_sz];

    let (_success, _residual, _iters) = minres_black_box(
        &mut sol,
        &black_box,
        &b,
        &shift,
        &tol,
        &max_iters,
        &mut plugin,
        &mut aux1,
        &mut aux2,
        &mut aux3,
        &mut aux4,
        &mut aux5,
    );

    println!("Primal:");
    for val in sol[0..var_sz].iter() {
        println!("{val:.7e}");
    }

    println!("Dual:");
    for val in sol[var_sz..sol.len()].iter() {
        println!("{val:.7e}");
    }
}
