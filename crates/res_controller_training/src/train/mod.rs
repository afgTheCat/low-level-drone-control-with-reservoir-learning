pub mod esn;
pub mod izhikevich;

use crate::utils::snapshots_to_motor_inputs;
use loaders::FlDataSet;
use res_controller::controllers::esn::{DroneRCParameters, NonAdaptingDroneRc};

pub fn train_on_dataset(
    dataset: &FlDataSet,
    training_parameters: DroneRCParameters,
) -> NonAdaptingDroneRc {
    let train_motor_input = snapshots_to_motor_inputs(&dataset.train_data);
    NonAdaptingDroneRc::train_new(&dataset.train_data, train_motor_input, training_parameters)
}
