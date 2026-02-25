use crate::input_mapping::ReservoirInput;
use flight_controller::{Channels, FlightController, FlightControllerUpdate, MotorInput};
use nalgebra::{DMatrix, DVector};
use res::izhikevich::{IzhikevichInput, IzhikevichReservoir};
use ridge::RidgeRegression;
use serde::{Deserialize, Serialize};
use std::{sync::Mutex, time::Duration};

#[derive(Serialize, Deserialize)]
pub struct IzhikevichController {
    pub reservoir: Mutex<IzhikevichReservoir>,
    pub spike_traces: Mutex<DVector<f64>>,
    // let dt_ms = reservoir.dt.as_secs_f64() * 1000.0;
    // let alpha = (-dt_ms / tau_ms).exp();
    pub spike_trace_decay_factor: f64,
    pub w_in: DMatrix<f64>,
    pub readout: RidgeRegression,
    pub scheduler_delta: Duration,
    pub network_delta: Duration,
    pub g_in: f64,
}

impl IzhikevichController {
    // something
    pub fn step(&self, input: IzhikevichInput) {
        let mut reservoir = self.reservoir.lock().unwrap();
        let mut spike_traces = self.spike_traces.lock().unwrap();
        let IzhikevichInput { input, duration } = input;
        let mut t = Duration::ZERO;
        while t < duration {
            let (input_and_firings, firings) = reservoir.diffuse(input.clone());
            *spike_traces *= self.spike_trace_decay_factor;
            for i in firings {
                // debug check in case something goes out of bounds
                spike_traces[i] += 1.0;
            }
            // self.update_spike_traces(firings);
            reservoir.excite(input_and_firings);
            t += reservoir.dt;
        }
    }

    pub fn izhikevich_input_calc(&self, update: ReservoirInput) -> IzhikevichInput {
        let fl_update_input = update.to_vector();
        let input = (&self.w_in * fl_update_input) * self.g_in;
        IzhikevichInput {
            input,
            duration: self.network_delta,
        }
    }

    fn representation(&self, update: FlightControllerUpdate) -> DVector<f64> {
        let spike_traces = self.spike_traces.lock().unwrap();
        let n = spike_traces.len();
        let mut out = DVector::zeros(1 + n + 4);

        out[0] = 1.0;
        out.rows_mut(1, n).copy_from(&spike_traces);
        let Channels {
            throttle,
            roll,
            pitch,
            yaw,
        } = update.channels;
        out[1 + n] = throttle;
        out[1 + n + 1] = yaw;
        out[1 + n + 2] = pitch;
        out[1 + n + 3] = roll;

        out
    }
}

impl FlightController for IzhikevichController {
    fn init(&self) {}

    fn update(&self, _delta_time: f64, update: FlightControllerUpdate) -> MotorInput {
        // NOTE: transform the update to an izhikevich reservoir input
        let reservoir_input = ReservoirInput::from_flight_controller_update(update);
        let izhikevich_input = self.izhikevich_input_calc(reservoir_input);
        self.step(izhikevich_input);
        let representation = self.representation(update);
        let pr = self.readout.predict(DMatrix::from_row_slice(
            1,
            representation.len(),
            representation.as_slice(),
        ));
        let motor_input_1 = f64::clamp(*pr.row(0).get(0).unwrap(), 0., 1.);
        let motor_input_2 = f64::clamp(*pr.row(0).get(1).unwrap(), 0., 1.);
        let motor_input_3 = f64::clamp(*pr.row(0).get(2).unwrap(), 0., 1.);
        let motor_input_4 = f64::clamp(*pr.row(0).get(3).unwrap(), 0., 1.);
        MotorInput {
            input: [motor_input_1, motor_input_2, motor_input_3, motor_input_4],
        }
    }

    fn scheduler_delta(&self) -> std::time::Duration {
        self.scheduler_delta
    }
}
