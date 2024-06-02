/// Computes the sparse matrix vector (SpMV) multiplication for f64.
/// Matrix should be in COO format, ideally sorted by rows.
pub fn spmv<'a>(
    product: &'a mut [f64],
    mrows: &[usize],
    mcols: &[usize],
    mvals: &[f64],
    vector: &[f64],
) -> &'a [f64] {
    product.reset();

    for (idx, m) in mvals.iter().enumerate() {
        product[mrows[idx]] += m * vector[mcols[idx]];
    }

    product
}

/// Computes the dot product between two vectors of type f64.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut product: f64 = 0.0;
    for ii in 0..x.len() {
        product += x[ii] * y[ii];
    }

    product
}

pub fn swap(x: &mut [f64], y: &mut [f64], spare: &mut [f64]) {
    spare.copy_from_slice(x);
    x.copy_from_slice(y);
    y.copy_from_slice(spare);
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
            *elm = beta * *elm;
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
