// use crate::RcInput;
use nalgebra::{Complex, ComplexField, DMatrix};
use rand::thread_rng;
use rand_distr::{Bernoulli, Distribution, Uniform};
use serde::{Deserialize, Serialize};

// classical ESN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Esn {
    pub n_internal_units: usize,
    pub input_scaling: f64,
    pub internal_weights: DMatrix<f64>,
    pub input_weights: DMatrix<f64>,
}

impl Esn {
    pub fn new(
        n_internal_units: usize,
        connectivity: f64,
        spectral_radius: f64,
        input_scaling: f64,
        n_vars: usize,
    ) -> Self {
        let internal_weights =
            Self::internal_weights(n_internal_units, connectivity, spectral_radius);
        let input_weights = Self::input_weights(n_internal_units, n_vars, input_scaling);
        Self {
            input_scaling,
            n_internal_units,
            internal_weights,
            input_weights,
        }
    }

    fn internal_weights(
        n_internal_units: usize,
        connectivity: f64,
        spectral_radius: f64,
    ) -> DMatrix<f64> {
        assert!(
            connectivity > 0.0 && connectivity <= 1.0,
            "Connectivity must be in (0, 1]."
        );

        // Generate a random sparse matrix with connectivity
        let mut rng = thread_rng();
        let uniform_dist = Uniform::new(-0.5, 0.5);
        let bernoulli = Bernoulli::new(connectivity).unwrap();
        let mut internal_weights = DMatrix::from_fn(n_internal_units, n_internal_units, |_, _| {
            if bernoulli.sample(&mut rng) {
                uniform_dist.sample(&mut rng)
            } else {
                0.0
            }
        });

        // Compute eigenvalues to find the spectral radius
        // let eigenvalues = internal_weights.clone().eigenvalues().unwrap();
        let eigenvalues = internal_weights.clone().schur().complex_eigenvalues();
        let max_eigenvalue = eigenvalues
            .iter()
            .cloned()
            .map(Complex::abs)
            .fold(f64::NEG_INFINITY, f64::max);

        // Scale matrix to match the desired spectral radius
        internal_weights /= max_eigenvalue / spectral_radius;

        internal_weights
    }

    fn input_weights(
        n_internal_units: usize,
        variables: usize,
        input_scaling: f64,
    ) -> DMatrix<f64> {
        let mut rng = thread_rng();
        let bernoulli = Bernoulli::new(0.5).unwrap();
        DMatrix::from_fn(n_internal_units, variables, |_, _| {
            if bernoulli.sample(&mut rng) {
                input_scaling
            } else {
                -input_scaling
            }
        })
    }

    pub fn integrate(
        &self,
        current_input: &DMatrix<f64>,
        previous_state: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let state_before_tanh = &self.internal_weights * previous_state.transpose()
            + &self.input_weights * current_input.transpose();
        state_before_tanh.map(|e| e.tanh()).transpose()
    }

    pub fn advance_state(&self, current_input: &DMatrix<f64>, previous_state: &mut DMatrix<f64>) {
        assert_eq!(
            current_input.nrows(),
            previous_state.nrows(),
            "advance_state: batch size mismatch"
        );
        assert_eq!(
            previous_state.ncols(),
            self.n_internal_units,
            "advance_state: state width mismatch"
        );
        *previous_state = self.integrate(current_input, previous_state);
    }

    // computes the strate matricies for each episode
    // episodes are represented as DMatrix where each row t represents the states at that time
    // pub fn compute_state_matricies(&self, input: &Box<dyn RcInput>) -> Vec<DMatrix<f64>> {
    //     let (eps, time, _) = input.shape();
    //     let n_internal_units = self.n_internal_units;
    //     let mut states: Vec<DMatrix<f64>> = vec![DMatrix::zeros(time, n_internal_units); eps];
    //     let mut previous_state: DMatrix<f64> = DMatrix::zeros(eps, n_internal_units);
    //
    //     for t in 0..time {
    //         let current_input = input.input_at_time(t);
    //         previous_state = self.integrate(&current_input, &previous_state);
    //
    //         for (ep, state) in states.iter_mut().enumerate() {
    //             state.set_row(t, &previous_state.row(ep));
    //         }
    //     }
    //
    //     states
    // }

    // NOTE: The input vector holds the inputs at index t in inputs[i]
    // The rows of input matrix the rows, and the columns are the separate reservoirs
    // returns the the vector of state trajectories associated with episodes
    pub fn compute_state_matricies2(
        &self,
        eps: usize,
        time: usize,
        inputs: Vec<DMatrix<f64>>,
    ) -> Vec<DMatrix<f64>> {
        let mut states: Vec<DMatrix<f64>> = vec![DMatrix::zeros(time, self.n_internal_units); eps];
        // eps rows, n_internal_units cols
        let mut previous_state: DMatrix<f64> = DMatrix::zeros(eps, self.n_internal_units);
        for (t, current_input) in inputs.into_iter().enumerate() {
            previous_state = self.integrate(&current_input, &previous_state);
            for (ep, state) in states.iter_mut().enumerate() {
                state.set_row(t, &previous_state.row(ep));
            }
        }
        states
    }
}
