// NOTE: these are the results that are going to be used!

use drone::default_drone::default_7in_4s_drone;
use flight_controller::Channels;
use flight_controller::FlightController;
use loaders::LoaderTrait;
use loaders::file_loader::FileLoader;
use loggers::memory_logger::MemoryLogger;
use rayon::prelude::*;
use res_controller::controllers::esn::DroneRCParameters;
use res_controller::dimensionality_reducer::ReducerType;
use res_controller_training::eval::angular_rate_stabilization::angular_rate_stabilization_test;
use res_controller_training::eval::angular_rate_stabilization::dummy_stabilization_db;
use res_controller_training::eval::open_loop_imitation_mse::OpenLoopEvaluationResult;
use res_controller_training::train::esn::train_and_evaluate_open_loop_imitation_mse;
use res_controller_training::train::train_on_dataset;
use res_controller_training::utils::append_result;
use res_controller_training::utils::results_path;
use sim_context::SimContext;
use simulator::Simulator;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

#[derive(Debug, Clone)]
struct ParameterSweep {
    base: DroneRCParameters,
    buffer_sizes: Vec<usize>,
    pca_dims: Vec<usize>,
}

impl ParameterSweep {
    fn drone_rc_parameters(&self) -> Vec<DroneRCParameters> {
        let mut out = Vec::with_capacity(self.buffer_sizes.len() * self.pca_dims.len());
        for &buffer_size in &self.buffer_sizes {
            for &pca_dim in &self.pca_dims {
                out.push(DroneRCParameters {
                    internal_units: self.base.internal_units,
                    connectivity: self.base.connectivity,
                    spectral_radius: self.base.spectral_radius,
                    input_scaling: self.base.input_scaling,
                    buffer_size,
                    alpha: self.base.alpha,
                    reducer_type: ReducerType::PCA(pca_dim),
                    use_setpoint_repr: self.base.use_setpoint_repr,
                });
            }
        }
        out
    }
}

fn esn_parameters_sweep_test_on_db(
    dataset_id: &str,
) -> Vec<((OpenLoopEvaluationResult, DroneRCParameters), String)> {
    let base = DroneRCParameters {
        internal_units: 200,
        connectivity: 0.15,
        spectral_radius: 0.9,
        input_scaling: 0.15,
        buffer_size: 1,
        alpha: 1.0,
        reducer_type: ReducerType::PCA(64), // will be overridden by sweep
        use_setpoint_repr: false,
    };
    let parameter_sweep = ParameterSweep {
        base,
        buffer_sizes: vec![1, 2, 4, 8],
        pca_dims: vec![16, 32, 64, 128],
    };

    let sweep = parameter_sweep.drone_rc_parameters();
    let blocks = sweep
        .into_par_iter()
        .map(|params| {
            let mut sim_context = SimContext::default();
            sim_context.set_loader(&sim_context::LoaderType::File);
            (
                train_and_evaluate_open_loop_imitation_mse(&mut sim_context, dataset_id, params),
                dataset_id.to_owned(),
            )
        })
        .collect::<Vec<_>>();
    blocks
}

#[test]
fn esn_parameter_sweep_test() {
    let tries_per_ds = 3;
    let dataset_ids = vec!["5_len_inc"];
    let file_name = results_path(&format!("combined5.csv"));
    let blocks: Vec<_> = dataset_ids
        .into_par_iter()
        .flat_map(|ds_id| {
            (0..tries_per_ds)
                .into_par_iter()
                .flat_map(|_| esn_parameters_sweep_test_on_db(ds_id))
                .collect::<Vec<_>>()
        })
        .collect();
    for ((eval_res, params), ds_id) in blocks {
        let DroneRCParameters {
            buffer_size,
            reducer_type: ReducerType::PCA(pca),
            ..
        } = params
        else {
            unreachable!()
        };

        let mean_sqrs = eval_res
            .episode_mses
            .iter()
            .map(|(simulation_id, mse)| {
                format!("{ds_id},{simulation_id},{buffer_size},{pca},{mse}\n")
            })
            .collect::<Vec<_>>()
            .concat();
        println!("{}", mean_sqrs);

        append_result(&file_name, &mean_sqrs);
    }
}

#[test]
fn evaluate_esn_stabilization() {
    let drone_params = DroneRCParameters {
        internal_units: 200,
        connectivity: 0.15,
        spectral_radius: 0.9,
        input_scaling: 0.15,
        buffer_size: 8,
        alpha: 1.,
        reducer_type: ReducerType::PCA(64),
        use_setpoint_repr: false,
    };

    (0..10).into_par_iter().for_each(|_| {
        for ds in ["5_len", "10_len", "25_len", "50_len"] {
            let controller_id = format!("{ds}_trained_2");
            let mut sim_context = SimContext::default();
            sim_context.set_loader(&sim_context::LoaderType::File);
            let mut fl_data_set = sim_context.loader.lock().unwrap().load_data_set(ds);
            fl_data_set.downsample(Duration::from_millis(10));
            let controller = train_on_dataset(&fl_data_set, drone_params);
            sim_context.insert_drone_rc(&controller_id, controller.clone());
            controller.init();
            let db = dummy_stabilization_db();
            let drone = sim_context.load_drone().unwrap();
            angular_rate_stabilization_test(drone, Arc::new(controller), db, &controller_id);
        }
    });
}

#[test]
fn compare_it_with_original() {
    let mut sim_context = SimContext::default();
    sim_context.set_loader(&sim_context::LoaderType::File);
    let mut fl = sim_context
        .loader
        .lock()
        .unwrap()
        .load_flight_log("manual_flight");
    let downsample_len = Duration::from_millis(10);
    fl.downsample(downsample_len);
    let logger = Arc::new(Mutex::new(MemoryLogger::new("mem".into())));
    let controller = sim_context
        .loader
        .lock()
        .unwrap()
        .load_res_controller("10_long_trained");
    let controller = Arc::new(controller);
    let drone = default_7in_4s_drone();
    let mut simulator =
        Simulator::with_drone_and_controller_logger(drone, controller.clone(), logger.clone());
    simulator.init();
    for snapshot in fl.steps.iter() {
        simulator.simulate_delta(downsample_len, snapshot.channels);
    }
    let logger = logger.lock().unwrap();
    logger.write_to_file("replayed.csv", fl);
}

fn evaluate_stabilization_like_test(
    drone: drone::Drone,
    controller: Arc<dyn FlightController>,
    stick_input: Channels,
    target: res_controller::bf_rates::Rates,
    tolerance: f64,
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
        if logger.last_n_close_to_angular(100, target.yaw, target.pitch, target.roll, tolerance) {
            return true;
        }
    }
    false
}

fn stabilization_success_rate(
    drone: &drone::Drone,
    controller: &Arc<dyn FlightController>,
    db: &res_controller_training::utils::angular_rate_stabilization_db::AngularRateStabilizationDb,
    tolerance: f64,
) -> f64 {
    controller.init();
    let total = db.sticks_and_targets.len();
    let mut successes = 0usize;

    for entry in &db.sticks_and_targets {
        if evaluate_stabilization_like_test(
            drone.clone(),
            controller.clone(),
            entry.stick_input,
            entry.target,
            tolerance,
        ) {
            successes += 1;
        }
    }

    successes as f64 / total as f64
}

#[test]
fn benchmark_all_loaded_controllers_as_tolerance_halves() {
    let db = dummy_stabilization_db();
    let drone = default_7in_4s_drone();

    let mut file_loader = FileLoader::default();
    let controller_ids = file_loader.get_reservoir_controller_ids();

    let mut controllers: Vec<(String, Arc<dyn FlightController>)> = Vec::new();
    for controller_id in controller_ids {
        let Some(controller) = file_loader.try_load_res_controller(&controller_id) else {
            println!("Skipping unserializable controller: {controller_id}");
            continue;
        };
        controllers.push((controller_id, Arc::new(controller)));
    }
    assert!(
        !controllers.is_empty(),
        "No reservoir controllers could be loaded."
    );

    let halving_steps = 8usize;
    let drop_below_rate = 0.7;
    for step in 0..=halving_steps {
        let tolerance = 2.0 / (1u64 << step) as f64;

        println!("tolerance={tolerance} controllers={}", controllers.len());
        println!(
            "remaining_controllers={}",
            controllers
                .iter()
                .map(|(controller_id, _)| controller_id.as_str())
                .collect::<Vec<_>>()
                .join(",")
        );

        let results_chunks = controllers
            .par_chunks(3)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|(controller_id, controller)| {
                        let rate = stabilization_success_rate(&drone, controller, &db, tolerance);
                        (controller_id.clone(), controller.clone(), rate)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let results = results_chunks.into_iter().flatten().collect::<Vec<_>>();

        let mut best_rate = -1.0f64;
        let mut best_controller_id: Option<String> = None;
        let mut survivors: Vec<(String, Arc<dyn FlightController>)> = Vec::new();
        let mut dropped: Vec<(String, f64)> = Vec::new();

        for (controller_id, controller, rate) in results {
            println!("tolerance={tolerance} controller={controller_id} rate={rate}");

            if rate > best_rate {
                best_rate = rate;
                best_controller_id = Some(controller_id.clone());
            }

            if rate >= drop_below_rate {
                survivors.push((controller_id, controller));
            } else {
                dropped.push((controller_id, rate));
            }
        }

        let best_controller_id = best_controller_id.unwrap_or_else(|| "<none>".into());
        println!("BEST tolerance={tolerance} controller={best_controller_id} rate={best_rate}");
        println!(
            "dropped_controllers={}",
            dropped
                .iter()
                .map(|(controller_id, rate)| format!("{controller_id}:{rate:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        );

        controllers = survivors;
        println!(
            "surviving_controllers={}",
            controllers
                .iter()
                .map(|(controller_id, _)| controller_id.as_str())
                .collect::<Vec<_>>()
                .join(",")
        );
        if controllers.is_empty() {
            println!("No controllers remaining at tolerance={tolerance}");
            break;
        }
    }
}
