use crate::utils::snapshots_to_motor_inputs;
use loggers::FlightLog;
use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng, rngs::StdRng};
use res::izhikevich::{IzhikevichHarness, IzhikevichInput, random_izhikevich};
use res_controller::{
    controllers::izhikevich_controller::IzhikevichController, input_mapping::ReservoirInput,
};
use ridge::RidgeRegression;
use std::{sync::Mutex, time::Duration};

pub struct IzhikevichControllerParameters {
    pub n: usize,
    pub p: f64,
    pub excit_frac: f64,
    pub scheduler_delta: Duration,
    pub network_delta: Duration,
    pub g_in: f64,
    pub alpha: f64,
}

pub fn train_izhikevich_controller2(
    controller_params: IzhikevichControllerParameters,
    flight_logs: &[FlightLog],
) -> IzhikevichController {
    let IzhikevichControllerParameters {
        n,
        p,
        excit_frac,
        scheduler_delta,
        network_delta,
        g_in,
        alpha,
    } = controller_params;
    let reservoir = random_izhikevich(n, p, excit_frac);
    let dt_ms = reservoir.dt.as_secs_f64() * 1000.0;
    let tau_ms = 20.; // NOTE: wil need to tune this
    let spike_trace_decay_factor = (-dt_ms / tau_ms).exp();

    let motor_inputs = snapshots_to_motor_inputs(flight_logs);
    let total_steps: usize = flight_logs.iter().map(|fl| fl.steps.len()).sum();
    assert_eq!(
        motor_inputs.nrows(),
        total_steps,
        "motor input rows must match total steps"
    );

    let input_dim = ReservoirInput::NVARS;

    let mut rng = StdRng::seed_from_u64(1337);
    let mut w_in = DMatrix::<f64>::zeros(n, input_dim);
    for i in 0..n {
        for j in 0..input_dim {
            // Uniform in [-1, 1]; you can switch to Normal(0,1) if you prefer.
            w_in[(i, j)] = rng.gen_range(-1.0..=1.0);
        }
    }

    let mut representation = DMatrix::zeros(total_steps, 1 + n + 4);
    for t in 0..total_steps {
        representation[(t, 0)] = 1.0;
    }

    let mut row = 0usize;
    for flight_log in flight_logs {
        let steps = &flight_log.steps;
        let inputs: Vec<_> = steps
            .iter()
            .map(|s| {
                let reservoir_input = ReservoirInput::from_snapshot(s);
                let i_in = (&w_in * reservoir_input.to_vector()) * g_in;
                IzhikevichInput {
                    input: i_in,
                    duration: network_delta,
                }
            })
            .collect();

        let mut harness = IzhikevichHarness::new(reservoir.clone());
        for input in inputs {
            harness.process_input(input);
        }

        let spike_traces = harness.calculate_spike_traces(spike_trace_decay_factor);
        assert_eq!(
            spike_traces.nrows(),
            steps.len(),
            "spike trace rows must match flight log steps"
        );

        representation
            .view_mut((row, 1), (steps.len(), n))
            .copy_from(&spike_traces);
        for (t, step) in steps.iter().enumerate() {
            representation[(row + t, 1 + n)] = step.channels.throttle;
            representation[(row + t, 1 + n + 1)] = step.channels.yaw;
            representation[(row + t, 1 + n + 2)] = step.channels.pitch;
            representation[(row + t, 1 + n + 3)] = step.channels.roll;
        }
        row += steps.len();
    }
    debug_assert_eq!(row, total_steps);

    let readout = RidgeRegression::fit_multiple_svd(alpha, representation, &motor_inputs);

    IzhikevichController {
        reservoir: Mutex::new(reservoir),
        spike_traces: Mutex::new(DVector::zeros(n)),
        spike_trace_decay_factor,
        w_in,
        readout,
        scheduler_delta,
        network_delta,
        g_in,
    }
}

#[cfg(test)]
mod test {
    use crate::{
        eval::{
            angular_rate_stabilization::{angular_rate_stabilization_test, dummy_stabilization_db},
            open_loop_imitation_mse::evaluate_open_loop_dataset_mse,
        },
        train::izhikevich::{IzhikevichControllerParameters, train_izhikevich_controller2},
    };
    use sim_context::SimContext;
    use std::{sync::Arc, time::Duration};

    #[test]
    fn train_and_evaluate_izhikevich_controller() {
        let mut sim_context = SimContext::default();
        sim_context.set_loader(&sim_context::LoaderType::File);

        let mut fl_data_set = sim_context
            .loader
            .lock()
            .unwrap()
            .load_data_set("intermediate_dataset");
        fl_data_set.downsample(Duration::from_millis(10));
        let params = IzhikevichControllerParameters {
            n: 512,
            p: 0.05,
            excit_frac: 0.80,
            scheduler_delta: Duration::from_millis(10),
            network_delta: Duration::from_millis(10),
            g_in: 5.,
            alpha: 1.,
        };

        let drone = sim_context.load_drone().unwrap();
        let controller = train_izhikevich_controller2(params, &fl_data_set.train_data);
        let res = evaluate_open_loop_dataset_mse(&controller, &fl_data_set);
        println!("{:#?}", res);
        sim_context
            .loader
            .lock()
            .unwrap()
            .insert_izhikevich_controller("initial_izhikevich_controller", &controller);
        let db = dummy_stabilization_db();
        angular_rate_stabilization_test(drone, Arc::new(controller), db, "izh_controller");
    }
}
