pub mod esn_res;
pub mod repr;
pub mod setpoints;

use crate::{
    controllers::{
        esn::{
            esn_res::DroneEsn,
            repr::BufferedStates,
            setpoints::{setpoints_from_flight_logs, setpoints_from_flight_update},
        },
        hstack,
    },
    dimensionality_reducer::{Reducer, ReducerType},
    input_mapping::{MultipleReservoirInputTrajectory, ReservoirInput},
};
use flight_controller::{FlightController, FlightControllerUpdate, MotorInput};
use loggers::FlightLog;
use nalgebra::DMatrix;
use ridge::RidgeRegression;
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DroneRCParameters {
    pub internal_units: usize,
    pub connectivity: f64,
    pub spectral_radius: f64,
    pub input_scaling: f64,
    pub buffer_size: usize,
    pub alpha: f64,
    pub reducer_type: ReducerType,
    pub use_setpoint_repr: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NonAdaptingDroneRc {
    pub esn: DroneEsn,
    pub representation: BufferedStates,
    pub reducer: Reducer,
    // Could be changed to ElasticNetWrapper
    pub readout: RidgeRegression,
    pub use_setpoint_repr: bool,
    #[serde(skip)]
    pub runtime_state: Arc<Mutex<DMatrix<f64>>>,
}

impl NonAdaptingDroneRc {
    pub fn train_new(
        train_data: &[FlightLog],
        motor_inputs: DMatrix<f64>,
        parameters: DroneRCParameters,
    ) -> Self {
        let DroneRCParameters {
            internal_units,
            connectivity,
            spectral_radius,
            input_scaling,
            buffer_size,
            alpha,
            reducer_type,
            use_setpoint_repr,
        } = parameters;
        let esn = DroneEsn::new(internal_units, connectivity, spectral_radius, input_scaling);
        let representation = BufferedStates::new(buffer_size);
        let multi_reservoir_inputs = MultipleReservoirInputTrajectory::from_flight_logs(train_data);
        let res_states = esn.compute_state_matricies(multi_reservoir_inputs);
        // TODO: this is kinda stupid?
        let state_repr = representation.repr2(res_states);
        let reducer = Reducer::new(reducer_type, &state_repr);
        let mut input_repr = reducer.transform(state_repr);

        if use_setpoint_repr {
            let setpoints_repr = setpoints_from_flight_logs(train_data);
            input_repr = hstack(input_repr, setpoints_repr);
        }
        // TODO: readd this!
        let readout = RidgeRegression::fit_multiple_svd(alpha, input_repr, &motor_inputs);
        Self {
            esn,
            representation,
            readout,
            reducer,
            use_setpoint_repr,
            runtime_state: Arc::new(Mutex::new(DMatrix::zeros(1, internal_units))),
        }
    }
}

impl FlightController for NonAdaptingDroneRc {
    fn init(&self) {
        self.representation.reset_online();
        let mut state = self.runtime_state.lock().unwrap();
        *state = DMatrix::zeros(1, self.esn.n_internal_units());
    }

    fn update(&self, _delta_time: f64, update: FlightControllerUpdate) -> MotorInput {
        let rc_input = ReservoirInput::from_flight_controller_update(update);
        let mut current_res_state = self.runtime_state.lock().unwrap();
        self.esn.advance_state(&rc_input, &mut current_res_state);
        let state_repr = self.representation.repr_online_step(&current_res_state);
        let mut input_repr = self.reducer.transform(state_repr);
        if self.use_setpoint_repr {
            let setpoints_repr = setpoints_from_flight_update(&update);
            input_repr = hstack(input_repr, setpoints_repr);
        }
        let pr = self.readout.predict(input_repr);
        let motor_input_1 = f64::clamp(*pr.row(0).get(0).unwrap(), 0., 1.);
        let motor_input_2 = f64::clamp(*pr.row(0).get(1).unwrap(), 0., 1.);
        let motor_input_3 = f64::clamp(*pr.row(0).get(2).unwrap(), 0., 1.);
        let motor_input_4 = f64::clamp(*pr.row(0).get(3).unwrap(), 0., 1.);
        MotorInput {
            input: [motor_input_1, motor_input_2, motor_input_3, motor_input_4],
        }
    }

    fn scheduler_delta(&self) -> Duration {
        Duration::from_millis(10)
    }
}
