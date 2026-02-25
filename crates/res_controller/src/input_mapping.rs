use flight_controller::{Channels, FlightControllerUpdate};
use loggers::{FlightLog, SnapShot};
use macros::data_vars;
use nalgebra::{DMatrix, DVector};

use crate::bf_rates::{BetaflightRates, stick_inputs_to_targets};

#[data_vars(NVARS)]
pub struct ReservoirInput {
    throttle: f64,
    // roll: f64,
    // yaw: f64,
    // pitch: f64,
    roll_rate: f64,
    pitch_rate: f64,
    yaw_rate: f64,
    roll_target: f64,
    pitch_target: f64,
    yaw_target: f64,
    roll_err: f64,
    pitch_err: f64,
    yaw_err: f64,
}

impl ReservoirInput {
    pub fn from_snapshot(snapshot: &SnapShot) -> Self {
        let rate_scale = BetaflightRates::default().max_rate;
        let expected_rates = stick_inputs_to_targets(&snapshot.channels);
        let [ang_roll, ang_pitch, ang_yaw] = snapshot.gyro_update.angular_velocity;

        let roll_rate = ang_roll / rate_scale;
        let pitch_rate = ang_pitch / rate_scale;
        let yaw_rate = ang_yaw / rate_scale;

        let roll_target = expected_rates.roll / rate_scale;
        let pitch_target = expected_rates.pitch / rate_scale;
        let yaw_target = expected_rates.yaw / rate_scale;

        let roll_err = roll_target - roll_rate;
        let pitch_err = pitch_target - pitch_rate;
        let yaw_err = yaw_target - yaw_rate;

        Self {
            throttle: snapshot.channels.throttle,
            // roll: snapshot.channels.roll,
            // yaw: snapshot.channels.yaw,
            // pitch: snapshot.channels.pitch,
            roll_rate,
            pitch_rate,
            yaw_rate,
            roll_target,
            pitch_target,
            yaw_target,
            roll_err,
            pitch_err,
            yaw_err,
        }
    }

    pub fn from_flight_controller_update(update: FlightControllerUpdate) -> Self {
        let rate_scale = BetaflightRates::default().max_rate;
        let expected_rates = stick_inputs_to_targets(&update.channels);
        let Channels {
            throttle,
            // roll,
            // pitch,
            // yaw,
            ..
        } = update.channels;
        let [ang_roll, ang_pitch, ang_yaw] = update.gyro_update.angular_velocity;

        let roll_rate = ang_roll / rate_scale;
        let pitch_rate = ang_pitch / rate_scale;
        let yaw_rate = ang_yaw / rate_scale;

        let roll_target = expected_rates.roll / rate_scale;
        let pitch_target = expected_rates.pitch / rate_scale;
        let yaw_target = expected_rates.yaw / rate_scale;

        let roll_err = roll_target - roll_rate;
        let pitch_err = pitch_target - pitch_rate;
        let yaw_err = yaw_target - yaw_rate;

        Self {
            throttle,
            // roll,
            // yaw,
            // pitch,
            roll_rate,
            pitch_rate,
            yaw_rate,
            roll_target,
            pitch_target,
            yaw_target,
            roll_err,
            pitch_err,
            yaw_err,
        }
    }

    pub fn to_vector(&self) -> DVector<f64> {
        DVector::from_row_slice(&[
            self.throttle,
            // self.roll,
            // self.yaw,
            // self.pitch,
            self.roll_rate,
            self.pitch_rate,
            self.yaw_rate,
            self.roll_target,
            self.pitch_target,
            self.yaw_target,
            self.roll_err,
            self.pitch_err,
            self.yaw_err,
        ])
    }

    pub fn to_matrix(&self) -> DMatrix<f64> {
        let v = self.to_vector();
        DMatrix::from_row_slice(1, v.len(), v.as_slice())
    }
}

// for multiple concurrent reservoir at a fixed input
// each input belongs to a different episode
pub struct MultipleReservoirInput(Vec<ReservoirInput>);

impl MultipleReservoirInput {
    fn eps(&self) -> usize {
        self.0.len()
    }

    fn into_matrix(self) -> DMatrix<f64> {
        let nrows = self.0.len();
        let ncols = ReservoirInput::NVARS;
        let mut m: DMatrix<f64> = DMatrix::zeros(nrows, ncols);
        for (i, input) in self.0.into_iter().enumerate() {
            let v = input.to_vector();
            m.set_row(i, &v.transpose());
        }
        m
    }

    // snapshots
    fn from_snapshots(snapshots: &[&SnapShot]) -> Self {
        let inputs = snapshots
            .iter()
            .map(|input| ReservoirInput::from_snapshot(input))
            .collect();
        Self(inputs)
    }
}

pub struct MultipleReservoirInputTrajectory(Vec<MultipleReservoirInput>);

impl MultipleReservoirInputTrajectory {
    // eps, time, inputs
    pub fn to_reservoir_input(self) -> (usize, usize, Vec<DMatrix<f64>>) {
        let eps = self.0[0].eps();
        let t = self.0.len();
        let inputs = self.0.into_iter().map(|m| m.into_matrix()).collect();
        (eps, t, inputs)
    }

    pub fn from_flight_logs(flight_logs: &[FlightLog]) -> Self {
        let mut inputs = vec![];
        let total_t = flight_logs[0].steps.len();
        for t in 0..total_t {
            let snapshots_at_t = flight_logs
                .iter()
                .map(|fl| &fl.steps[t])
                .collect::<Vec<_>>();
            let multi_reservoir_input = MultipleReservoirInput::from_snapshots(&snapshots_at_t);
            inputs.push(multi_reservoir_input);
        }
        Self(inputs)
    }
}
