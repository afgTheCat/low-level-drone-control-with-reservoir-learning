pub mod default_laoder;
pub mod file_loader;

use drone::Drone;
use loggers::FlightLog;
use res_controller::controllers::{
    esn::NonAdaptingDroneRc, izhikevich_controller::IzhikevichController,
};
use std::time::Duration;

#[derive(Default, Debug)]
pub struct FlDataSet {
    pub dataset_id: String,
    pub train_data: Vec<FlightLog>,
    pub test_data: Vec<FlightLog>,
}

impl FlDataSet {
    pub fn downsample(&mut self, duration: Duration) {
        let FlDataSet {
            train_data,
            test_data,
            ..
        } = self;
        train_data.iter_mut().for_each(|d| d.downsample(duration));
        test_data.iter_mut().for_each(|d| d.downsample(duration));
    }
}

pub trait LoaderTrait: Send + Sync {
    // load a drone
    fn load_drone(&mut self, config_id: &str) -> Drone;

    // Load replay
    fn load_flight_log(&mut self, sim_id: &str) -> FlightLog;

    // Get simulation ids
    fn get_replay_ids(&mut self) -> Vec<String>;

    // Get reservoir ids
    fn get_reservoir_controller_ids(&mut self) -> Vec<String>;

    // Get Izhikevich controller ids
    fn get_izhikevich_controller_ids(&mut self) -> Vec<String>;

    // Insert a new reservoir
    fn insert_rc_controller(&mut self, controller_id: &str, controller: NonAdaptingDroneRc);

    // Load reservoir controller
    fn load_res_controller(&mut self, controller_id: &str) -> NonAdaptingDroneRc;

    fn load_data_set(&mut self, dataset_id: &str) -> FlDataSet;

    fn insert_data_set(&mut self, dataset: FlDataSet);

    fn insert_izhikevich_controller(
        &mut self,
        controller_id: &str,
        controller: &IzhikevichController,
    );

    fn load_izhikevich_controller(&mut self, controller_id: &str) -> IzhikevichController;
}
