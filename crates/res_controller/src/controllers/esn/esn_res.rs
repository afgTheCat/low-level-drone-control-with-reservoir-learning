use nalgebra::DMatrix;
use res::esn::Esn;
use serde::{Deserialize, Serialize};

use crate::input_mapping::{MultipleReservoirInputTrajectory, ReservoirInput};

#[derive(Clone, Serialize, Deserialize)]
pub struct DroneEsn(Esn);

impl DroneEsn {
    pub fn new(
        n_internal_units: usize,
        connectivity: f64,
        spectral_radius: f64,
        input_scaling: f64,
    ) -> Self {
        let esn = Esn::new(
            n_internal_units,
            connectivity,
            spectral_radius,
            input_scaling,
            ReservoirInput::NVARS,
        );
        Self(esn)
    }

    pub fn compute_state_matricies(
        &self,
        input: MultipleReservoirInputTrajectory,
    ) -> Vec<DMatrix<f64>> {
        let (eps, time, inputs) = input.to_reservoir_input();
        self.0.compute_state_matricies2(eps, time, inputs)
    }

    pub fn n_internal_units(&self) -> usize {
        self.0.n_internal_units
    }

    pub fn advance_state(&self, current_input: &ReservoirInput, previous_state: &mut DMatrix<f64>) {
        let current_input = current_input.to_matrix();
        self.0.advance_state(&current_input, previous_state);
    }
}
