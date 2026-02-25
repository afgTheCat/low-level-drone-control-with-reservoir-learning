use nalgebra::{DMatrix, DVector};

pub struct Rls {
    w: DMatrix<f64>,
    p: DMatrix<f64>,
    lambda: f64,
}

impl Rls {
    pub fn new(n: usize, m: usize, lambda: f64, p_init: f64) -> Self {
        Self {
            w: DMatrix::zeros(m, n),
            p: DMatrix::identity(n, n) * p_init,
            lambda,
        }
    }

    pub fn update(&mut self, x: &DVector<f64>, y: &DVector<f64>) {
        let px = &self.p * x;
        let denom = self.lambda + x.dot(&px);
        let k = px / denom;
        let y_hat = &self.w * x;
        let e = y - y_hat;
        self.w += &e * k.transpose();
        self.p = (&self.p - &k * x.transpose() * &self.p) / self.lambda;
    }

    pub fn predict(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.w * x
    }
}
