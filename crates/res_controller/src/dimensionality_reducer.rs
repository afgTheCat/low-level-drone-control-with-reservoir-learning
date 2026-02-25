use nalgebra::{DMatrix, DVector, SVD};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct PCADimensionalityReducer {
    mean: DVector<f64>,
    components: DMatrix<f64>,
    max_components: usize,
}

impl PCADimensionalityReducer {
    pub fn fit_with_max_components(input: &DMatrix<f64>, max_components: usize) -> Self {
        let (samples, features) = input.shape();
        if samples == 0 || features == 0 {
            return Self {
                mean: DVector::zeros(0),
                components: DMatrix::zeros(0, 0),
                max_components,
            };
        }

        let mean = DVector::from(
            input
                .column_iter()
                .map(|col| col.mean())
                .collect::<Vec<_>>(),
        );

        let mut centered = input.clone();
        for (i, mut col) in centered.column_iter_mut().enumerate() {
            col.add_scalar_mut(-mean[i]);
        }

        let SVD {
            u: _,
            v_t,
            singular_values: _,
        } = SVD::new(centered, false, true);
        let v_t = v_t.expect("SVD V^T missing");

        let k = max_components.min(features);
        let components = v_t.rows(0, k).transpose();

        Self {
            mean,
            components,
            max_components,
        }
    }

    pub fn transform(&self, input: DMatrix<f64>) -> DMatrix<f64> {
        let (samples, features) = input.shape();
        if samples == 0 || features == 0 {
            return input;
        }
        if self.components.ncols() == 0 {
            return input;
        }
        assert_eq!(
            features,
            self.mean.len(),
            "DimensionalityReducer: feature count mismatch"
        );
        assert_eq!(
            features,
            self.components.nrows(),
            "DimensionalityReducer: components row mismatch"
        );

        let mut centered = input;
        for (i, mut col) in centered.column_iter_mut().enumerate() {
            col.add_scalar_mut(-self.mean[i]);
        }

        centered * &self.components
    }
}

impl Default for PCADimensionalityReducer {
    fn default() -> Self {
        Self {
            mean: DVector::zeros(0),
            components: DMatrix::zeros(0, 0),
            max_components: 512,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NullReducer;

impl NullReducer {
    pub fn transform(&self, input: DMatrix<f64>) -> DMatrix<f64> {
        input
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ReducerType {
    PCA(usize),
    Null,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Reducer {
    PCAReducer(PCADimensionalityReducer),
    NullReducer(NullReducer),
}

impl Reducer {
    pub fn pca_reducer(input: &DMatrix<f64>, max_components: usize) -> Self {
        Self::PCAReducer(PCADimensionalityReducer::fit_with_max_components(
            input,
            max_components,
        ))
    }

    pub fn null_reducer() -> Self {
        Self::NullReducer(NullReducer)
    }

    pub fn new(reducer_type: ReducerType, input: &DMatrix<f64>) -> Self {
        match reducer_type {
            ReducerType::PCA(max_components) => Self::pca_reducer(input, max_components),
            ReducerType::Null => Self::null_reducer(),
        }
    }

    pub fn transform(&self, input: DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Self::PCAReducer(reducer) => reducer.transform(input),
            Self::NullReducer(reducer) => reducer.transform(input),
        }
    }
}
