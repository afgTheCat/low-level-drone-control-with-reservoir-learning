pub mod angular_rate_stabilization_db;

use flight_controller::Channels;
use loggers::FlightLog;
use nalgebra::{DMatrix, DVector};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

pub fn results_path(filename: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(filename)
}

pub fn reset_results_file(path: &Path, header: &str) {
    let mut file = File::create(path).expect("Failed to create results file");
    writeln!(file, "{header}").expect("Failed to write header");
}

pub fn append_result(path: &Path, content: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("Failed to open results file");
    writeln!(file, "{content}").expect("Failed to write result");
}

pub fn spectral_radius_linspace_inclusive(start: f64, end: f64, steps: usize) -> Vec<f64> {
    assert!(steps >= 2, "steps must be >= 2");
    let step = (end - start) / (steps - 1) as f64;
    (0..steps).map(|i| start + step * i as f64).collect()
}

pub fn generate_channel_commands() -> Vec<Channels> {
    let number_of_steps = 5; // -1, -0.5, 0., 0.5, 1.
    let step_size = 2. / (number_of_steps as f64 - 1.);
    let controller_values: Vec<_> = (0..number_of_steps)
        .map(|i| -1. + step_size * i as f64)
        .collect();
    let mut all_channel_targets: Vec<Channels> = vec![];
    for yaw in controller_values.iter() {
        for pitch in controller_values.iter() {
            for roll in controller_values.iter() {
                all_channel_targets.push(Channels {
                    throttle: -1.,
                    roll: *roll,
                    pitch: *pitch,
                    yaw: *yaw,
                });
            }
        }
    }
    all_channel_targets
}

pub fn snapshots_to_motor_inputs(flight_logs: &[FlightLog]) -> DMatrix<f64> {
    let mut dmatrix_columns = vec![];
    for flight_log in flight_logs {
        dmatrix_columns.extend(
            flight_log
                .steps
                .iter()
                .map(|e| DVector::from_row_slice(&e.motor_input.input)),
        )
    }
    DMatrix::from_columns(&dmatrix_columns).transpose()
}
