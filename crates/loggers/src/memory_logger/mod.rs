use crate::{FlightLog, Logger, SnapShot};
use std::fs::File;
use std::io::Write;
use std::{fs::OpenOptions, path::Path};

#[derive(Debug)]
pub struct MemoryLogger {
    pub simulation_id: String,
    pub snapshots: Vec<SnapShot>,
}

// const TARGET_ANGULAR_RATE_CLOSE: f64 = 2.;

impl MemoryLogger {
    pub fn new(simulation_id: String) -> Self {
        Self {
            simulation_id,
            snapshots: vec![],
        }
    }

    pub fn last_n_close_to_angular(
        &self,
        n: usize,
        target_yaw: f64,
        target_pitch: f64,
        target_roll: f64,
        target: f64,
    ) -> bool {
        if self.snapshots.len() < n {
            false
        } else {
            self.snapshots.iter().rev().take(n).all(|snap_shot| {
                let [pitch, yaw, roll] = snap_shot.gyro_update.angular_velocity;
                f64::abs(pitch - target_pitch) < target
                    && f64::abs(roll - target_roll) < target
                    && f64::abs(yaw - target_yaw) < target
            })
        }
    }

    pub fn last_n_yaw_avg(&self, n: usize) -> f64 {
        // println!("{}", self.snapshots.len());
        let yaw_sum: f64 = self
            .snapshots
            .iter()
            .rev()
            .take(n)
            .map(|s| {
                // println!("{:?}", s.gyro_update.angular_velocity);
                s.gyro_update.angular_velocity[1]
            })
            .sum();
        yaw_sum / n as f64
    }

    pub fn write_to_file(&self, filename: &str, fl: FlightLog) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(filename);
        let mut file: File = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("Failed to open results file");
        println!("{:?}", file);
        let snapshot_iter = self.snapshots.iter().zip(fl.steps);
        for (snapshot, step) in snapshot_iter {
            let angular_velocity = snapshot.gyro_update.angular_velocity;
            writeln!(
                file,
                "{}, {}, {}, {}, {}, {}",
                angular_velocity[0],
                angular_velocity[1],
                angular_velocity[2],
                step.gyro_update.angular_velocity[0],
                step.gyro_update.angular_velocity[1],
                step.gyro_update.angular_velocity[2]
            )
            .unwrap();
        }
    }
}

impl Logger for MemoryLogger {
    fn log_time_stamp(&mut self, snapshot: SnapShot) {
        self.snapshots.push(snapshot);
    }

    fn flush(&mut self) {
        self.snapshots.clear();
    }
}
