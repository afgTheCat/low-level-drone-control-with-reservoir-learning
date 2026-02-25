use flight_controller::FlightControllerUpdate;
use loggers::FlightLog;
use nalgebra::DMatrix;

pub fn setpoints_from_flight_logs(fls: &[FlightLog]) -> DMatrix<f64> {
    let len = 4;
    let total_rows: usize = fls.iter().map(|fl| fl.steps.len()).sum();
    let mut out = DMatrix::zeros(total_rows, len);
    let mut row = 0usize;
    for ep in fls {
        for snapshot in &ep.steps {
            out[(row, 0)] = snapshot.channels.throttle;
            out[(row, 1)] = snapshot.channels.roll;
            out[(row, 2)] = snapshot.channels.yaw;
            out[(row, 3)] = snapshot.channels.pitch;
            row += 1;
        }
    }
    out
}

pub fn setpoints_from_flight_update(fl: &FlightControllerUpdate) -> DMatrix<f64> {
    DMatrix::from_row_slice(
        1,
        4,
        &[
            fl.channels.throttle,
            fl.channels.roll,
            fl.channels.yaw,
            fl.channels.pitch,
        ],
    )
}
