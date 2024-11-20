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
    let tol: f64 = 1e-12;

    // Specify max iterations allowed
    let max_iters = sz.pow(2);

    // Select an appropriate plugin.
    let mut do_nothing = DoNothing::new();
    let mut console_logger = ConsoleLogger::extend(&mut do_nothing);
    let mut plugin = StopWatch::extend(&mut console_logger);

    // let mut plugin = StopWatch::extend(&mut do_nothing);

    // Initialize auxiliary vectors.
    let mut aux1: Vec<f64> = vec![0.0; sz];
    let mut aux2: Vec<f64> = vec![0.0; sz];
    let mut aux3: Vec<f64> = vec![0.0; sz];
    let mut aux4: Vec<f64> = vec![0.0; sz];
    let mut aux5: Vec<f64> = vec![0.0; sz];

    let mut average_solver_time: f64 = 0.0;

    let mut success: isize;

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

    println!(" ");
    println!("Top left 10x10 block of the matrix inverse:");
    println!(" ");
    for kk in 0..10 {
        for val in inverse_matrix[kk][0..10].iter() {
            write!(&mut f, "{val} ").ok();
            print!("{val: <+14.7e} ");
        }
        write!(&mut f, "\n").ok();
        print!("\n");
    }

    println!("Total solver executions: {sz}.");
    println!("Average solver execution time: {:.7e} s.", average_solver_time);
}
