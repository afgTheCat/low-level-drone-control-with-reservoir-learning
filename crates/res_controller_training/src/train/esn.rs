use crate::{
    eval::open_loop_imitation_mse::{OpenLoopEvaluationResult, evaluate_open_loop_dataset_mse},
    train::train_on_dataset,
};
use res_controller::controllers::esn::DroneRCParameters;
use sim_context::SimContext;
use std::time::Duration;

pub fn train_and_evaluate_open_loop_imitation_mse(
    sim_context: &mut SimContext,
    dataset_id: &str,
    training_parameters: DroneRCParameters,
) -> (OpenLoopEvaluationResult, DroneRCParameters) {
    let mut fl_data_set = sim_context.loader.lock().unwrap().load_data_set(dataset_id);
    fl_data_set.downsample(Duration::from_millis(10));

    let params_for_log = format!("{training_parameters:?}");

    let drone_rc = train_on_dataset(&fl_data_set, training_parameters);
    let evaluation_result = evaluate_open_loop_dataset_mse(&drone_rc, &fl_data_set);
    sim_context.insert_drone_rc(&params_for_log, drone_rc);
    (evaluation_result, training_parameters)
}
