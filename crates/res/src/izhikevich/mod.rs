use nalgebra::{clamp, dvector, DMatrix, DVector};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Serialize, Deserialize)]
pub struct IzhikevichReservoir {
    pub neuron_count: usize,
    pub a: DVector<f64>,
    pub b: DVector<f64>,
    pub c: DVector<f64>,
    pub d: DVector<f64>,
    pub connections: DMatrix<f64>,
    // states
    pub v: DVector<f64>,
    pub u: DVector<f64>,
    pub dt: Duration,
}

// we should build around this!
impl IzhikevichReservoir {
    pub fn new(
        a: DVector<f64>,
        b: DVector<f64>,
        c: DVector<f64>,
        d: DVector<f64>,
        connections: DMatrix<f64>,
        v: DVector<f64>,
        u: DVector<f64>,
    ) -> Self {
        let neuron_count = a.iter().count();
        Self {
            neuron_count,
            a,
            b,
            c,
            d,
            connections,
            v,
            u,
            dt: Duration::from_millis(1),
        }
    }

    pub fn diffuse(&mut self, mut input: DVector<f64>) -> (DVector<f64>, Vec<usize>) {
        let mut firings = vec![];
        for neuron_id in 0..self.neuron_count {
            let neuron_v = self.v[neuron_id];
            if neuron_v > 30. {
                firings.push(neuron_id);
            }
        }
        for firing_neuron in firings.iter() {
            self.v[*firing_neuron] = self.c[*firing_neuron];
            self.u[*firing_neuron] += self.d[*firing_neuron];
            input += self.connections.column(*firing_neuron);
        }
        (input, firings)
    }

    pub fn excite(&mut self, input: DVector<f64>) {
        let dt = self.dt.as_secs_f64() * 1000.;
        let input = input.map(|i| clamp(i, -100., 50.));
        let f = |v: f64, u: f64, i: f64| 0.04 * v * v + 5.0 * v + 140.0 - u + i;

        // Two half steps for v (Izhikevich's recommended scheme)
        let half_dt = 0.5 * dt;

        self.v += half_dt * self.v.zip_zip_map(&self.u, &input, f);
        self.v += half_dt * self.v.zip_zip_map(&self.u, &input, f);

        // Full step for u using updated v
        let x = self.v.zip_zip_map(&self.b, &self.u, |v, b, u| b * v - u);
        self.u += dt * self.a.zip_map(&x, |a, x| a * x);
    }
}

pub struct IzhikevichInput {
    pub input: DVector<f64>,
    pub duration: Duration,
}

pub struct IzhikevichHarness {
    pub reservoir: IzhikevichReservoir,
    // log the firings
    pub firings: Vec<Vec<usize>>,
    // cumulative end indices into `firings` for each processed input window
    pub input_end_steps: Vec<usize>,
    // log the total time spent
    pub total_time: Duration,
}

impl IzhikevichHarness {
    pub fn new(reservoir: IzhikevichReservoir) -> Self {
        Self {
            reservoir,
            firings: vec![],
            input_end_steps: vec![],
            total_time: Duration::ZERO,
        }
    }

    pub fn process_input(&mut self, input: IzhikevichInput) {
        let IzhikevichInput { input, duration } = input;
        let mut t = Duration::ZERO;
        while t < duration {
            let (input_and_firings, firings) = self.reservoir.diffuse(input.clone());
            self.firings.push(firings);
            self.reservoir.excite(input_and_firings);
            t += self.reservoir.dt;
        }
        self.total_time += t;
        self.input_end_steps.push(self.firings.len());
    }

    pub fn print_stats(&self) {
        let neuron_count = self.reservoir.neuron_count;
        let dt_secs = self.total_time.as_secs_f64();
        let spikes_t: Vec<usize> = self.firings.iter().map(|f| f.len()).collect();
        if spikes_t.is_empty() {
            println!("No timesteps. spikes_t is empty.");
            return;
        }

        let t_steps = spikes_t.len();
        let total_spikes: usize = spikes_t.iter().sum();

        // Mean spikes per timestep
        let mean = total_spikes as f64 / t_steps as f64;

        // Std dev of spikes per timestep
        let var = spikes_t
            .iter()
            .map(|&s| {
                let d = s as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / t_steps as f64;

        let std = var.sqrt();

        // Coefficient of variation (burstiness proxy)
        let cv = if mean > 1e-12 { std / mean } else { 0.0 };

        // Burst classification threshold:
        // - default: mean + 2*std
        // - if std ~ 0, fallback to something meaningful
        let threshold = if std > 1e-12 {
            mean + 2.0 * std
        } else {
            // If activity is perfectly flat, treat "burst" as significantly above mean.
            // This is somewhat arbitrary; adjust if needed.
            mean + 1.0
        };

        let burst_steps = spikes_t.iter().filter(|&&s| (s as f64) > threshold).count();
        let burst_frac = burst_steps as f64 / t_steps as f64;

        // Optional: normalized population spike rate (spikes / neuron / second)
        // NOTE: This uses TOTAL spikes, not "active reservoirs".
        // let rate_hz = total_spikes as f64 / (neuron_count as f64 * (t_steps as f64 * dt_secs));
        let total_time_secs = self.total_time.as_secs_f64();
        let rate_hz = total_spikes as f64 / (neuron_count as f64 * total_time_secs);

        println!("--- Reservoir activity (Test 2: burstiness/synchrony) ---");
        println!("timesteps: {t_steps}, dt: {dt_secs}s");
        println!("total spikes: {total_spikes}");
        println!("rate_hz (spikes/neuron/sec): {rate_hz:.4}");
        println!("spikes_t mean: {mean:.4}, std: {std:.4}, CV: {cv:.4}");
        println!(
            "burst threshold: {:.4} spikes/step, burst steps: {}, burst fraction: {:.4}",
            threshold, burst_steps, burst_frac
        );

        // A tiny extra sanity signal: how often is the reservoir completely silent?
        let silent_steps = spikes_t.iter().filter(|&&s| s == 0).count();
        println!(
            "silent fraction: {:.4}",
            silent_steps as f64 / t_steps as f64
        );

        let rate_hz = spikes_t.iter().sum::<usize>() as f64 / (neuron_count as f64 * dt_secs);
        println!("rate hz: {rate_hz}");
    }

    pub fn calculate_spike_traces(&self, decay_factor: f64) -> DMatrix<f64> {
        let n = self.reservoir.neuron_count;
        if n == 0 {
            return DMatrix::zeros(self.input_end_steps.len(), 0);
        }
        if self.input_end_steps.is_empty() {
            return DMatrix::zeros(0, n);
        }

        let mut traces = DVector::<f64>::zeros(n);
        let mut rows: Vec<DVector<f64>> = Vec::with_capacity(self.input_end_steps.len());

        let mut next_end_idx = 0usize;
        let mut current_end = self.input_end_steps[next_end_idx];
        for (t_idx, firings_t) in self.firings.iter().enumerate() {
            traces *= decay_factor;
            for &i in firings_t {
                traces[i] += 1.0;
            }

            if t_idx + 1 == current_end {
                rows.push(traces.clone());
                next_end_idx += 1;
                if next_end_idx >= self.input_end_steps.len() {
                    break;
                }
                current_end = self.input_end_steps[next_end_idx];
            }
        }

        let rows = rows.into_iter().map(|v| v.transpose()).collect::<Vec<_>>();
        DMatrix::from_rows(&rows)
    }
}

pub fn regular_spiking_single_neuron_reservoir() -> IzhikevichReservoir {
    let a = dvector![0.02];
    let b = dvector![0.2];
    let c = dvector![-65.];
    let d = dvector![8.];
    let v0 = dvector![-65.];
    let u0 = dvector![0.2 * -65.];
    IzhikevichReservoir::new(a, b, c, d, DMatrix::zeros(1, 1), v0, u0)
}

pub fn fast_spiking_single_neuron_reservoir() -> IzhikevichReservoir {
    let a = dvector![0.1];
    let b = dvector![0.2];
    let c = dvector![-65.];
    let d = dvector![2.];
    let v0 = dvector![-65.];
    let u0 = dvector![0.2 * -65.];
    IzhikevichReservoir::new(a, b, c, d, DMatrix::zeros(1, 1), v0, u0)
}

pub fn random_izhikevich(n: usize, p: f64, excit_frac: f64) -> IzhikevichReservoir {
    let mut rng = StdRng::seed_from_u64(42);

    // Weight magnitudes (baseline; you’ll likely add a global g_rec later)
    let w_e_max: f64 = 0.5; // excitatory weights ~ U(0, w_e_max)
    let w_i_max: f64 = 1.0; // inhibitory weights ~ U(-w_i_max, 0)

    // Izhikevich params
    // Excitatory RS: a=0.02, b=0.2, c=-65, d=8
    // Inhibitory FS: a=0.1,  b=0.2, c=-65, d=2
    let e_count = ((n as f64) * excit_frac).round() as usize;

    let mut a = DVector::<f64>::zeros(n);
    let mut b = DVector::<f64>::zeros(n);
    let mut c = DVector::<f64>::zeros(n);
    let mut d = DVector::<f64>::zeros(n);

    for i in 0..n {
        if i < e_count {
            a[i] = 0.02;
            b[i] = 0.2;
            c[i] = -65.0;
            d[i] = 8.0;
        } else {
            a[i] = 0.1;
            b[i] = 0.2;
            c[i] = -65.0;
            d[i] = 2.0;
        }
    }

    let mut v = DVector::<f64>::zeros(n);
    let mut u = DVector::<f64>::zeros(n);

    for i in 0..n {
        // Start near resting potential with small noise
        let noise = rng.gen_range(-5.0..=5.0);
        v[i] = -65.0 + noise;
        u[i] = b[i] * v[i];
    }

    let mut connections = DMatrix::<f64>::zeros(n, n);
    for pre in 0..n {
        let is_excit = pre < e_count;
        for post in 0..n {
            if post == pre {
                continue; // optional: no self-connection
            }
            if rng.r#gen::<f64>() < p {
                let w = if is_excit {
                    rng.gen_range(0.0..=w_e_max)
                } else {
                    -rng.gen_range(0.0..=w_i_max)
                };
                connections[(post, pre)] = w;
            }
        }
    }
    IzhikevichReservoir::new(a, b, c, d, connections, v, u)
}
