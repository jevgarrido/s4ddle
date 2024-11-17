use num_traits::ToPrimitive;
use s4ddle::*;
use s4ddle::plugins::*;
use s4ddle::readwrite::*;
use s4ddle::operators::*;

use std::io::Write;

fn main() {
    // Load subject marix to be inverted.
    let (rows, cols, vals, sz, _nonzeros) = mm_read("./data/ash85.csv");
    // let (rows, cols, vals, sz, _nonzeros) = mm_read("./data/nos4.csv");

    // Initialize dense solution matrix
    let mut inverse_matrix: Vec<Vec<f64>> = vec![vec![0.0; sz]; sz];

    // Initialize right hand side
    let mut b: Vec<f64> = vec![0.0; sz];

    // Specify diagonal shift
    let shift: f64 = 0.0;

    // Specify tolerance
    let tol: f64 = 1e-15;

    // Specify max iterations allowed
    let max_iters = sz.pow(2);

    // Select an appropriate plugin.
    // let mut plugin = DoNothing::new();
    let mut plugin = StopWatch::new();

    // Initialize auxiliary vectors.
    let mut aux1: Vec<f64> = vec![0.0; sz];
    let mut aux2: Vec<f64> = vec![0.0; sz];
    let mut aux3: Vec<f64> = vec![0.0; sz];
    let mut aux4: Vec<f64> = vec![0.0; sz];
    let mut aux5: Vec<f64> = vec![0.0; sz];

    let mut average_solver_time: f64 = 0.0;

    let mut success: isize;

    let mut outer_loop_timer = StopWatch::new();

    <StopWatch as Plugin<f64>>::start(&mut outer_loop_timer);

    for kk in 0..sz {
        b.reset();
        b[kk] = 1.0;

        (success, _, _) = minres(
            &mut inverse_matrix[kk],
            &rows,
            &cols,
            &vals,
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

        if success != 1 {
            println!("Failed to obtain solution at iteration {kk}");
        }

        average_solver_time = (plugin.elapsed_time_in_seconds() + average_solver_time * (kk.to_f64().unwrap() + 1.0))
            / (kk.to_f64().unwrap() + 2.0);
    }

    <StopWatch as Plugin<f64>>::end(&mut outer_loop_timer);

    // ----
    // Write the first (top left) 10x10 block of the matrix inverse to a csv file.

    let f = std::fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open("matrix_inverse.csv")
        .ok()
        .unwrap();
    let mut f = std::io::BufWriter::new(f);

    for kk in 0..10 {
        for val in inverse_matrix[kk][0..10].iter() {
            write!(&mut f, "{val} ").ok();
        }
        write!(&mut f, "\n").ok();
    }

    println!("Success! ^^");
    println!("Total solver executions: {sz}.");
    println!("Average solver execution time: {:.7e} s.", average_solver_time);
    println!("Total elapsed time: {:.7e} s.", outer_loop_timer.elapsed_time_in_seconds());
}
