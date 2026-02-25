use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct FlightInput {
    pub episodes: usize,
    pub time: usize,
    pub vars: usize, // I guess this can be something else
    pub reservoir_input: Vec<DMatrix<f64>>,
}

impl FlightInput {
    pub fn new_from_rc_input(flight_logs: Vec<Vec<DVector<f64>>>) -> Self {
        let episodes = flight_logs.len();
        let time = flight_logs.iter().map(|x| x.len()).max().unwrap_or(0);
        let data = flight_logs
            .iter()
            .map(|fl| {
                // let columns = fl.iter().map(db_fl_to_rc_input).collect::<Vec<_>>();
                DMatrix::from_columns(fl).transpose()
            })
            .collect();
        let vars = flight_logs
            .iter()
            .find_map(|ep| ep.first())
            .map(|v| v.len())
            .unwrap_or(0);
        Self {
            episodes,
            time,
            vars,
            reservoir_input: data,
        }
    }
}
