use std::time::Duration;

use crate::{Logger, SnapShot};
use rerun::RecordingStream;

pub struct RerunLogger {
    rec: RecordingStream,
}

impl Logger for RerunLogger {
    fn log_time_stamp(&mut self, snapshot: SnapShot) {
        self.rec.set_time("stable_time", snapshot.duration);
        // let position = snapshot.current_frame.drone_frame_state.position;
        // let point = Vec3::new(position.x as f32, position.z as f32, position.y as f32);
        // let points = Points3D::new(vec![point]);
        // self.rec
        //     .log("drone/pos", &points.with_radii([0.8]))
        //     .unwrap();
    }

    fn flush(&mut self) {
        self.rec.flush_blocking();
    }
}

impl Drop for RerunLogger {
    fn drop(&mut self) {
        self.flush();
    }
}

impl RerunLogger {
    pub fn new(simulation_id: String) -> Self {
        let rec = rerun::RecordingStreamBuilder::new(simulation_id)
            .spawn()
            .unwrap();
        rec.set_time("stable_time", Duration::ZERO);

        Self { rec }
    }
}
