use loggers::FlightLog;
use nalgebra::DMatrix;
use rand::{Rng, SeedableRng, rngs::StdRng};
use res::izhikevich::{IzhikevichHarness, IzhikevichInput, random_izhikevich};
use res_controller::input_mapping::ReservoirInput;
use sim_context::SimContext;
use std::time::Duration;

fn transform_data_set(
    flight_log: FlightLog,
    number_of_neurons: usize,
    sample_rate: Duration,
) -> Vec<IzhikevichInput> {
    let FlightLog { steps, .. } = flight_log;
    let res_inputs: Vec<_> = steps
        .iter()
        .map(|s| {
            let reservoir_input = ReservoirInput::from_snapshot(s);
            reservoir_input.to_vector()
        })
        .collect();
    let input_dim = res_inputs[0].len();
    let mut rng = StdRng::seed_from_u64(1337);
    let mut w_in = DMatrix::<f64>::zeros(number_of_neurons, input_dim);
    for i in 0..number_of_neurons {
        for j in 0..input_dim {
            // Uniform in [-1, 1]; you can switch to Normal(0,1) if you prefer.
            w_in[(i, j)] = rng.gen_range(-1.0..=1.0);
        }
    }
    let g_in: f64 = 5.0;
    res_inputs
        .into_iter()
        .map(|res_input| {
            let i_in = (w_in.clone() * res_input) * g_in;
            IzhikevichInput {
                input: i_in,
                duration: sample_rate,
            }
        })
        .collect()
}

#[test]
fn find_izhikevich_reservoir() {
    let sample_rate = Duration::from_millis(10);

    // Load data
    let mut sim_context = SimContext::default();
    sim_context.set_loader(&sim_context::LoaderType::File);
    let mut fl_data_set = sim_context.loader.lock().unwrap().load_data_set("10_len");
    fl_data_set.downsample(sample_rate);

    // Load the testing harness
    let reservoir = random_izhikevich(512, 0.05, 0.80);
    let mut harness = IzhikevichHarness::new(reservoir);

    let first_flight = fl_data_set.train_data[0].clone();
    let inputs = transform_data_set(first_flight, harness.reservoir.neuron_count, sample_rate);
    for input in inputs {
        harness.process_input(input);
    }
    harness.print_stats();
}
