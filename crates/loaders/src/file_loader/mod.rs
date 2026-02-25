use drone::Drone;
use res_controller::controllers::{
    esn::NonAdaptingDroneRc, izhikevich_controller::IzhikevichController,
};
use std::{fs, path::PathBuf};

use crate::{FlDataSet, LoaderTrait};

pub fn loader_path() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/quad")
}

#[derive(Debug, Default)]
pub struct FileLoader {}

impl LoaderTrait for FileLoader {
    fn load_drone(&mut self, config_id: &str) -> Drone {
        let mut drone_path = loader_path();
        drone_path.push("drones/");
        fs::create_dir_all(&drone_path).unwrap();
        drone_path.push(format!("{config_id}.json"));
        let content = fs::read_to_string(drone_path).unwrap();
        serde_json::from_slice(content.as_bytes()).unwrap()
    }

    fn load_flight_log(&mut self, sim_id: &str) -> loggers::FlightLog {
        let mut replay = loader_path();
        replay.push("replays/");
        fs::create_dir_all(&replay).unwrap();
        replay.push(sim_id);
        let content = fs::read_to_string(replay).unwrap();
        serde_json::from_slice(content.as_bytes()).unwrap()
    }

    // just list the file names in the loader.
    fn get_replay_ids(&mut self) -> Vec<String> {
        let mut replays_dir = loader_path();
        replays_dir.push("replays/");
        fs::create_dir_all(&replays_dir).unwrap();
        fs::read_dir(replays_dir)
            .unwrap()
            .map(|res| res.map(|e| e.path()))
            .filter_map(|res| res.ok().map(|t| t.to_str().unwrap().to_owned()))
            .collect::<Vec<_>>()
    }

    fn get_reservoir_controller_ids(&mut self) -> Vec<String> {
        let mut reservoir_dir = loader_path();
        reservoir_dir.push("reservoirs/");
        fs::create_dir_all(&reservoir_dir).unwrap();
        fs::read_dir(reservoir_dir)
            .unwrap()
            .map(|res| res.map(|e| e.path()))
            .filter_map(|res| res.ok().map(|t| t.to_str().unwrap().to_owned()))
            .collect::<Vec<_>>()
    }

    fn get_izhikevich_controller_ids(&mut self) -> Vec<String> {
        let mut controller_dir = loader_path();
        controller_dir.push("izhikevich_controllers/");
        fs::create_dir_all(&controller_dir).unwrap();
        fs::read_dir(controller_dir)
            .unwrap()
            .map(|res| res.map(|e| e.path()))
            .filter_map(|res| res.ok().map(|t| t.to_str().unwrap().to_owned()))
            .collect::<Vec<_>>()
    }

    fn insert_rc_controller(&mut self, controller_id: &str, controller: NonAdaptingDroneRc) {
        let mut reservoir_dir = loader_path();
        reservoir_dir.push("reservoirs/");
        fs::create_dir_all(&reservoir_dir).unwrap();
        reservoir_dir.push(controller_id);
        let serialized = serde_json::to_string(&controller).unwrap();
        fs::write(reservoir_dir, serialized).unwrap();
    }

    fn load_res_controller(&mut self, controller_id: &str) -> NonAdaptingDroneRc {
        let mut reservoir_dir = loader_path();
        reservoir_dir.push("reservoirs/");
        fs::create_dir_all(&reservoir_dir).unwrap();
        reservoir_dir.push(controller_id);
        let content = fs::read_to_string(reservoir_dir).unwrap();
        serde_json::from_slice(content.as_bytes()).unwrap()
    }

    fn load_data_set(&mut self, dataset_id: &str) -> FlDataSet {
        let mut dataset_dir = loader_path();
        dataset_dir.push(format!("replays/{dataset_id}/"));
        fs::create_dir_all(&dataset_dir).unwrap();
        let entries = fs::read_dir(dataset_dir)
            .unwrap()
            .map(|res| res.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let train_set = entries
            .iter()
            .filter(|e| {
                // file name
                let filename = e.file_name().unwrap();
                filename
                    .as_encoded_bytes()
                    .starts_with("training_".as_bytes())
            })
            .collect::<Vec<_>>();
        let test_set = entries
            .iter()
            .filter(|e| {
                let filename = e.file_name().unwrap();
                filename
                    .as_encoded_bytes()
                    .starts_with("testing_".as_bytes())
            })
            .collect::<Vec<_>>();
        let train_data = train_set
            .iter()
            .map(|path| {
                let content = fs::read_to_string(path).unwrap();
                serde_json::from_slice(content.as_bytes()).unwrap()
            })
            .collect();
        let test_data = test_set
            .iter()
            .map(|path| {
                let content = fs::read_to_string(path).unwrap();
                serde_json::from_slice(content.as_bytes()).unwrap()
            })
            .collect();
        FlDataSet {
            dataset_id: dataset_id.into(),
            train_data,
            test_data,
        }
    }

    fn insert_data_set(&mut self, dataset: FlDataSet) {
        let FlDataSet {
            dataset_id,
            train_data,
            test_data,
        } = dataset;
        let mut dataset_dir = loader_path();
        dataset_dir.push(format!("replays/{dataset_id}/"));
        fs::create_dir_all(&dataset_dir).unwrap();
        for fl in train_data {
            let mut fl_path = dataset_dir.clone();
            fl_path.push(&fl.simulation_id);
            let contents = serde_json::to_string(&fl).unwrap();
            fs::write(fl_path, contents).unwrap();
        }
        for fl in test_data {
            let mut fl_path = dataset_dir.clone();
            fl_path.push(&fl.simulation_id);
            let contents = serde_json::to_string(&fl).unwrap();
            fs::write(fl_path, contents).unwrap();
        }
    }

    fn insert_izhikevich_controller(
        &mut self,
        controller_id: &str,
        controller: &IzhikevichController,
    ) {
        let mut controller_path = loader_path();
        controller_path.push("izhikevich_controllers/");
        fs::create_dir_all(&controller_path).unwrap();
        controller_path.push(controller_id);
        let serialized = serde_json::to_string(controller).unwrap();
        fs::write(controller_path, serialized).unwrap();
    }

    fn load_izhikevich_controller(&mut self, controller_id: &str) -> IzhikevichController {
        let mut controller_path = loader_path();
        controller_path.push("izhikevich_controllers/");
        fs::create_dir_all(&controller_path).unwrap();
        controller_path.push(controller_id);
        let content = fs::read_to_string(controller_path).unwrap();
        serde_json::from_slice(content.as_bytes()).unwrap()
    }
}

impl FileLoader {
    pub fn try_load_res_controller(&mut self, controller_id: &str) -> Option<NonAdaptingDroneRc> {
        let mut reservoir_dir = loader_path();
        reservoir_dir.push("reservoirs/");
        fs::create_dir_all(&reservoir_dir).ok()?;
        reservoir_dir.push(controller_id);
        let content = fs::read_to_string(reservoir_dir).ok()?;
        serde_json::from_slice(content.as_bytes()).ok()
    }
}

#[cfg(test)]
mod test {
    use crate::file_loader::loader_path;
    use drone::default_drone::default_7in_4s_drone;
    use std::fs;

    #[test]
    fn save_default_config_to_file() {
        let default_drone = default_7in_4s_drone();
        let serialized = serde_json::to_string(&default_drone).unwrap();
        let mut drone_path = loader_path();
        drone_path.push("drones");
        fs::create_dir_all(&drone_path).unwrap();
        drone_path.push("7in_4s_drone.json");
        fs::write(drone_path, serialized).unwrap();
    }
}
