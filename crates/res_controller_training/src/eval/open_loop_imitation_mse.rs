use flight_controller::FlightController;
use flight_controller::FlightControllerUpdate;
use loaders::FlDataSet;
use loggers::FlightLog;
use std::time::Duration;

#[derive(Debug)]
pub struct OpenLoopEvaluationResult {
    pub mean_mse: f64,
    pub episode_mses: Vec<(String, f64)>,
}

pub fn evaluate_open_loop_fl_mse<C: FlightController>(
    controller: &C,
    flight_log: &FlightLog,
) -> f64 {
    let mut predicted_motor_inputs = vec![];
    let mut actual_motor_inputs = vec![];
    for snapshot in flight_log.steps.iter() {
        actual_motor_inputs.push(snapshot.motor_input);
        let duration = Duration::default();
        let update = FlightControllerUpdate {
            battery_update: snapshot.battery_update,
            gyro_update: snapshot.gyro_update,
            channels: snapshot.channels,
        };
        let prediction = controller.update(duration.as_secs_f64(), update);
        predicted_motor_inputs.push(prediction);
    }

    let total_samples = (predicted_motor_inputs.len() * 4) as f64;
    let mut squared_error_sum = 0.0;
    for (predicted, actual) in predicted_motor_inputs
        .iter()
        .zip(actual_motor_inputs.iter())
    {
        for i in 0..4 {
            let diff = predicted[i] - actual[i];
            squared_error_sum += diff * diff;
        }
    }

    squared_error_sum / total_samples
}

pub fn evaluate_open_loop_dataset_mse<C: FlightController>(
    controller: &C,
    dataset: &FlDataSet,
) -> OpenLoopEvaluationResult {
    let episode_mses = dataset
        .test_data
        .iter()
        .map(|flight_log| {
            let simulation_id = flight_log.simulation_id.clone();
            let mse = evaluate_open_loop_fl_mse(controller, flight_log);
            (simulation_id, mse)
        })
        .collect::<Vec<_>>();
    let mean_mse = episode_mses.iter().map(|(_, mse)| mse).sum::<f64>() / episode_mses.len() as f64;
    OpenLoopEvaluationResult {
        mean_mse,
        episode_mses,
    }
}
