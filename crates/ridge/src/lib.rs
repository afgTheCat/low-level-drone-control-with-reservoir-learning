pub mod ridge2;

use nalgebra::{DMatrix, DVector, RawStorage, SVD};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeRegressionSol {
    pub coeff: DMatrix<f64>,
    pub intercept: DVector<f64>,
}

// Super basic ridge regression model using svd
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeRegression {
    pub alpha: f64,
    pub sol: Option<RidgeRegressionSol>,
}

impl RidgeRegression {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, sol: None }
    }

    pub fn fit_on_decomposed_svd(
        u: &DMatrix<f64>,
        v_t: &DMatrix<f64>,
        d: &DVector<f64>,
        mut y: DVector<f64>,
        x_mean: &DVector<f64>,
    ) -> (DVector<f64>, f64) {
        let y_mean = y.mean();
        for elem in y.iter_mut() {
            *elem -= y_mean
        }

        let coeff = v_t.transpose() * d.zip_map(&(u.transpose() * y), |x, y| x * y);
        let intercept = y_mean - x_mean.dot(&coeff);

        (coeff, intercept)
    }

    pub fn fit_multiple_svd(alpha: f64, mut x: DMatrix<f64>, y: &DMatrix<f64>) -> Self {
        let (_, data_features) = x.data.shape();
        let (_, target_dim) = y.data.shape();
        let mut coeff_mult: DMatrix<f64> = DMatrix::zeros(target_dim.0, data_features.0);
        let mut intercept_mult: DVector<f64> = DVector::zeros(target_dim.0);

        let x_mean = DVector::from(x.column_iter().map(|col| col.mean()).collect::<Vec<_>>());
        for (i, mut col) in x.column_iter_mut().enumerate() {
            col.add_scalar_mut(-x_mean[i]);
        }

        let SVD {
            u,
            v_t,
            singular_values,
        } = SVD::new(x, true, true);

        let d = singular_values.map(|sig| sig / (sig.powi(2) + alpha));
        let v_t = v_t.unwrap();
        let u = u.unwrap();

        for (i, col) in y.column_iter().enumerate() {
            let (coeff, intercept) = Self::fit_on_decomposed_svd(&u, &v_t, &d, col.into(), &x_mean);
            coeff_mult.set_row(i, &coeff.transpose());
            intercept_mult[i] = intercept;
        }
        Self {
            alpha,
            sol: Some(RidgeRegressionSol {
                coeff: coeff_mult.clone(),
                intercept: intercept_mult.clone(),
            }),
        }
    }

    // makes a prediction, based on the calculated solution!
    pub fn predict(&self, repr: DMatrix<f64>) -> DMatrix<f64> {
        let Some(RidgeRegressionSol { coeff, intercept }) = self.sol.as_ref() else {
            panic!()
        };

        DMatrix::from_rows(
            &repr
                .row_iter()
                .map(|data_point_repr| {
                    DVector::from(
                        intercept
                            .iter()
                            .zip(coeff.row_iter())
                            .map(|(ic, coeff_row)| ic + data_point_repr.dot(&coeff_row))
                            .collect::<Vec<_>>(),
                    )
                    .transpose()
                })
                .collect::<Vec<_>>(),
        )
    }
}
