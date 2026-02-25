use flight_controller::Channels;
use loaders::file_loader::loader_path;
use res_controller::bf_rates::Rates;
use serde::{Deserialize, Serialize};
use std::{fs, io::Read};

#[derive(Debug, Serialize, Deserialize)]
pub struct StickAndTarget {
    pub stick_input: Channels,
    pub target: Rates,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AngularRateStabilizationDb {
    pub db_name: String,
    pub sticks_and_targets: Vec<StickAndTarget>,
}

pub fn list_angular_rate_stabilization_dbs() -> Vec<String> {
    let mut angular_rate_stabilization_dbs_path = loader_path();
    angular_rate_stabilization_dbs_path.push("angular_rate_dbs");
    fs::create_dir_all(&angular_rate_stabilization_dbs_path).unwrap();
    let entries = fs::read_dir(angular_rate_stabilization_dbs_path)
        .unwrap()
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    entries
        .iter()
        .map(|entry| {
            let filename = entry.file_name().unwrap();
            let mut buf = String::new();
            filename
                .as_encoded_bytes()
                .read_to_string(&mut buf)
                .unwrap();
            buf
        })
        .collect()
}

pub fn load_rate_stabilization_db(db_name: &str) -> AngularRateStabilizationDb {
    let mut angular_rate_stabilization_dbs_path = loader_path();
    angular_rate_stabilization_dbs_path.push("angular_rate_dbs");
    fs::create_dir_all(&angular_rate_stabilization_dbs_path).unwrap();
    let entries = fs::read_dir(angular_rate_stabilization_dbs_path)
        .unwrap()
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    entries
        .iter()
        .find_map(|entry| {
            let filename = entry.file_name().unwrap();
            if filename.as_encoded_bytes() == db_name.as_bytes() {
                let content = fs::read_to_string(entry).unwrap();
                Some(serde_json::from_slice(content.as_bytes()).unwrap())
            } else {
                None
            }
        })
        .unwrap()
}
