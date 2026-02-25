// TODO: needs refactor! will do it some day

use nalgebra::DMatrix;
use nalgebra::RowDVector;
// use res::RcInput;
use serde::Deserialize;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BufferedStates {
    pub lag: usize,
    #[serde(skip, default = "default_online_buffer")]
    online_buffer: Arc<Mutex<VecDeque<RowDVector<f64>>>>,
}

fn default_online_buffer() -> Arc<Mutex<VecDeque<RowDVector<f64>>>> {
    Arc::new(Mutex::new(VecDeque::new()))
}

impl Clone for BufferedStates {
    fn clone(&self) -> Self {
        Self::new(self.lag)
    }
}

impl BufferedStates {
    pub fn new(lag: usize) -> Self {
        Self {
            lag,
            online_buffer: default_online_buffer(),
        }
    }

    pub fn reset_online(&self) {
        self.online_buffer.lock().unwrap().clear();
    }

    // TODO: check this! I think this might be bad!
    pub fn repr2(&self, res_states: Vec<DMatrix<f64>>) -> DMatrix<f64> {
        let lag = self.lag;
        let eps = res_states.len();
        let mut features_all_eps = Vec::new();

        for states in res_states.iter().take(eps) {
            let time = states.nrows();
            let n_units = states.ncols();
            // why is there a +1?
            let mut features = DMatrix::zeros(time, n_units * (lag + 1));
            for t in 0..time {
                for k in 0..=lag {
                    let src_t = t.saturating_sub(k);
                    let dst_start = k * n_units;

                    features
                        .view_mut((t, dst_start), (1, n_units))
                        .copy_from(&states.view((src_t, 0), (1, n_units)));
                }
            }
            features_all_eps.push(features);
        }

        let mut all_rows: Vec<RowDVector<f64>> = Vec::new();

        for episode_matrix in features_all_eps {
            for row in episode_matrix.row_iter() {
                all_rows.push(row.into());
            }
        }

        DMatrix::from_rows(&all_rows)
    }

    pub fn repr_online_step(&self, current_res_state: &DMatrix<f64>) -> DMatrix<f64> {
        let lag = self.lag;
        let n_units = current_res_state.ncols();
        let current_row: RowDVector<f64> = current_res_state.row(0).into_owned();

        let mut buf = self.online_buffer.lock().unwrap();
        if let Some(existing) = buf.front() {
            // TODO: I think this should be cleared instead
            if existing.len() != n_units {
                buf.clear();
            }
        }

        buf.push_front(current_row);
        while buf.len() > lag + 1 {
            buf.pop_back();
        }

        let oldest = buf
            .back()
            .expect("online buffer must contain at least one element");

        let mut out = vec![0.0f64; n_units * (lag + 1)];
        for k in 0..=lag {
            let state_k = buf.get(k).unwrap_or(oldest);
            let dst = k * n_units;
            out[dst..dst + n_units].copy_from_slice(state_k.as_slice());
        }

        DMatrix::from_row_slice(1, out.len(), &out)
    }
}
