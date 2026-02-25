// NOTE: only loads the default drone config. This is for debugging and stuff

use drone::default_drone::default_7in_4s_drone;
use res_controller::controllers::{
    esn::NonAdaptingDroneRc, izhikevich_controller::IzhikevichController,
};

use crate::{FlDataSet, LoaderTrait};

#[derive(Debug, Default)]
pub struct DefaultLoader {}

impl LoaderTrait for DefaultLoader {
    fn load_drone(&mut self, _config_id: &str) -> drone::Drone {
        default_7in_4s_drone()
    }

    fn load_flight_log(&mut self, _sim_id: &str) -> loggers::FlightLog {
        unreachable!()
    }

    fn get_replay_ids(&mut self) -> Vec<String> {
        vec![]
    }

    fn get_reservoir_controller_ids(&mut self) -> Vec<String> {
        vec![]
    }

    fn get_izhikevich_controller_ids(&mut self) -> Vec<String> {
        vec![]
    }

    fn load_res_controller(&mut self, _controller_id: &str) -> NonAdaptingDroneRc {
        unreachable!()
    }

    fn insert_rc_controller(&mut self, _controller_id: &str, _controller: NonAdaptingDroneRc) {
        todo!()
    }

    fn load_data_set(&mut self, _dataset_id: &str) -> FlDataSet {
        FlDataSet::default()
    }

    fn insert_data_set(&mut self, _dataset: FlDataSet) {}

    fn insert_izhikevich_controller(
        &mut self,
        _controller_id: &str,
        _controller: &IzhikevichController,
    ) {
        todo!()
    }

    fn load_izhikevich_controller(&mut self, _controller_id: &str) -> IzhikevichController {
        todo!()
    }
}
