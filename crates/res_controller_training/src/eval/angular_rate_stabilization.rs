use crate::utils::angular_rate_stabilization_db::{AngularRateStabilizationDb, StickAndTarget};
use drone::Drone;
use flight_controller::{Channels, FlightController};
use loggers::memory_logger::MemoryLogger;
use rand::{Rng, SeedableRng, rngs::StdRng};
use res_controller::bf_rates::{Rates, stick_inputs_to_targets};
use simulator::Simulator;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

fn evaluate_stabilization_test(
    drone: Drone,
    controller: Arc<dyn FlightController>,
    stick_input: Channels,
    target: Rates,
) -> bool {
    let controller_delta = controller.scheduler_delta();
    let logger = Arc::new(Mutex::new(MemoryLogger::new("test".into())));
    let mut t = Duration::ZERO;
    let mut simulator =
        Simulator::with_drone_and_controller_logger(drone, controller, logger.clone());
    while t < Duration::from_secs(2) {
        t += controller_delta;
        simulator.simulate_delta(controller_delta, stick_input);
        let logger = logger.lock().unwrap();
        if logger.last_n_close_to_angular(100, target.yaw, target.pitch, target.roll, 2.) {
            println!("stabalized after {t:?}");
            return true;
        }
    }
    false
}

pub fn angular_rate_stabilization_test(
    drone: Drone,
    controller: Arc<dyn FlightController>,
    eval_db: AngularRateStabilizationDb,
    controller_id: &str,
) -> f64 {
    let mut successes = 0;
    let total_evals = eval_db.sticks_and_targets.len();
    for StickAndTarget {
        stick_input,
        target,
    } in eval_db.sticks_and_targets
    {
        if evaluate_stabilization_test(drone.clone(), controller.clone(), stick_input, target) {
            successes += 1;
        }
    }

    println!(
        "Controller {controller_id}, Total stabilization rate: {}",
        successes as f64 / total_evals as f64
    );

    successes as f64 / total_evals as f64
}

pub fn dummy_stabilization_db() -> AngularRateStabilizationDb {
    let throttle = 0.0_f64;
    let mut rng = StdRng::seed_from_u64(42);

    let n = 20; // bump this up/down
    let mut sticks_and_targets = Vec::with_capacity(n);

    for _ in 0..n {
        // Small bias toward “near zero” + occasional bigger commands
        let roll = sample_stick(&mut rng);
        let pitch = sample_stick(&mut rng);
        let yaw = sample_stick(&mut rng);

        let stick_input = Channels {
            throttle,
            roll,
            pitch,
            yaw,
        };
        sticks_and_targets.push(make_entry(stick_input));
    }

    AngularRateStabilizationDb {
        db_name: "test_db".into(),
        sticks_and_targets,
    }
}

fn sample_stick(rng: &mut StdRng) -> f64 {
    // 70% small range, 30% full range
    if rng.r#gen::<f64>() < 0.7 {
        rng.gen_range(-0.3..0.3)
    } else {
        rng.gen_range(-1.0..1.0)
    }
}

fn make_entry(stick_input: Channels) -> StickAndTarget {
    let target = stick_inputs_to_targets(&stick_input);
    StickAndTarget {
        stick_input,
        target,
    }
}
