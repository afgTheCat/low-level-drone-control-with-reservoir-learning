pub mod input_gen;

use bf_controller::BFController;
use drone::Drone;
use flight_controller::{controllers::null_controller::NullController, FlightController};
use loaders::LoaderTrait;
use loaders::{default_laoder::DefaultLoader, file_loader::FileLoader};
use loggers::{
    empty_logger::EmptyLogger, file_logger::FileLogger, rerun_logger::RerunLogger,
    Logger as LoggerTrait,
};
use loggers::{FlightLog, Logger};
use res_controller::controllers::esn::NonAdaptingDroneRc;
use simulator::Replayer;
use simulator::Simulator;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Default, Eq, PartialEq, Hash, Debug, Clone)]
pub enum LoggerType {
    File(String),
    Rerun(String),
    #[default]
    Empty,
}

#[derive(Clone, Eq, PartialEq, Hash, Debug, Default)]
pub enum ControllerType {
    #[default]
    Betafligt, // no parameters
    Reservoir(String),  // reservoir controller id
    Izhikevich(String), // izhikevich controller id
    NullController,     // no controller
}

#[derive(Debug)]
pub enum Loader {
    FileLoader(FileLoader),
    DefaultLoader(DefaultLoader),
}

impl Loader {
    pub fn load_drone(&mut self, config_id: &str) -> Drone {
        match self {
            Self::FileLoader(loader) => loader.load_drone(config_id),
            Self::DefaultLoader(loader) => loader.load_drone(config_id),
        }
    }

    pub fn load_res_controller(&mut self, controller_id: &str) -> NonAdaptingDroneRc {
        match self {
            Self::FileLoader(loader) => loader.load_res_controller(controller_id),
            Self::DefaultLoader(loader) => loader.load_res_controller(controller_id),
        }
    }

    pub fn load_replay(&mut self, replay_id: &str) -> FlightLog {
        match self {
            Self::FileLoader(loader) => loader.load_flight_log(replay_id),
            Self::DefaultLoader(loader) => loader.load_flight_log(replay_id),
        }
    }
}

impl Default for Loader {
    fn default() -> Self {
        Self::FileLoader(FileLoader::default())
    }
}

#[derive(Default, Clone, PartialEq)]
pub enum LoaderType {
    File,
    #[default]
    DefaultLoader,
}

pub struct SimContext {
    // Logger
    pub flight_controller: Arc<dyn FlightController>,
    // Controller
    pub logger: Arc<Mutex<dyn Logger>>,
    // Loader
    pub loader: Arc<Mutex<dyn LoaderTrait>>,

    // Replay ids
    pub replay_ids: Vec<String>,
    // Res controller ids
    pub reservoir_controller_ids: Vec<String>,
    // Izhikevich controller ids
    pub izhikevich_controller_ids: Vec<String>,
    // Selected replay id
    pub replay_id: Option<String>,
    // Config id
    pub config_id: Option<String>,
}

impl std::fmt::Debug for SimContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimContext")
            .field("replay_ids", &self.replay_ids)
            .field("reservoir_controller_ids", &self.reservoir_controller_ids)
            .field("izhikevich_controller_ids", &self.izhikevich_controller_ids)
            .field("replay_ids", &self.replay_ids)
            .field("config_id", &self.config_id)
            .finish()
    }
}

impl Default for SimContext {
    fn default() -> Self {
        let mut sim_context = SimContext {
            logger: Arc::new(Mutex::new(EmptyLogger::default())),
            flight_controller: Arc::new(NullController::default()),
            loader: Arc::new(Mutex::new(DefaultLoader::default())),
            replay_ids: Default::default(),
            reservoir_controller_ids: Default::default(),
            izhikevich_controller_ids: Default::default(),
            replay_id: Default::default(),
            config_id: Some("7in_4s_drone".into()),
        };
        sim_context.refresh_cache();
        sim_context
    }
}

impl SimContext {
    pub fn set_loader(&mut self, loader_type: &LoaderType) {
        match loader_type {
            LoaderType::File => self.loader = Arc::new(Mutex::new(FileLoader::default())),
            LoaderType::DefaultLoader => {
                self.loader = Arc::new(Mutex::new(DefaultLoader::default()))
            }
        }
    }

    pub fn set_replay_id(&mut self, replay_id: String) {
        self.replay_id = Some(replay_id)
    }

    pub fn set_logger(&mut self, logger_type: LoggerType) {
        let logger: Arc<Mutex<dyn LoggerTrait>> = match logger_type {
            LoggerType::Rerun(log_id) => Arc::new(Mutex::new(RerunLogger::new(log_id))),
            LoggerType::Empty => Arc::new(Mutex::new(EmptyLogger::default())),
            LoggerType::File(log_id) => Arc::new(Mutex::new(FileLogger::new(log_id))),
        };
        self.logger = logger;
    }

    pub fn set_controller(&mut self, controller: ControllerType) {
        let flight_controller: Arc<dyn FlightController> = match controller {
            ControllerType::Betafligt => Arc::new(BFController::default()),
            ControllerType::Reservoir(res_id) => {
                let res_controller = self.loader.lock().unwrap().load_res_controller(&res_id);
                Arc::new(res_controller)
            }
            ControllerType::Izhikevich(controller_id) => {
                let controller = self
                    .loader
                    .lock()
                    .unwrap()
                    .load_izhikevich_controller(&controller_id);
                Arc::new(controller)
            }
            ControllerType::NullController => Arc::new(NullController::default()),
        };
        self.flight_controller = flight_controller;
    }

    pub fn load_simulator(&self, config_id: &str) -> Simulator {
        let drone = self.loader.lock().unwrap().load_drone(config_id);
        Simulator::with_drone_and_controller_logger(
            drone,
            self.flight_controller.clone(),
            self.logger.clone(),
        )
    }

    pub fn try_load_simulator(&mut self) -> Option<Simulator> {
        let config_id = self.config_id.clone()?;
        Some(self.load_simulator(&config_id))
    }

    pub fn load_replay_ids(&mut self) {
        let replay_ids = self.loader.lock().unwrap().get_replay_ids();
        self.replay_ids = replay_ids
    }

    pub fn load_res_controllers_ids(&mut self) {
        let reservoir_controler_ids = self.loader.lock().unwrap().get_reservoir_controller_ids();
        self.reservoir_controller_ids = reservoir_controler_ids
    }

    pub fn load_izhikevich_controllers_ids(&mut self) {
        let controller_ids = self.loader.lock().unwrap().get_izhikevich_controller_ids();
        self.izhikevich_controller_ids = controller_ids
    }

    pub fn refresh_cache(&mut self) {
        self.load_replay_ids();
        self.load_res_controllers_ids();
        self.load_izhikevich_controllers_ids();
    }

    pub fn load_flight_log(&mut self, replay_id: &str) -> FlightLog {
        self.loader.lock().unwrap().load_flight_log(replay_id)
    }

    pub fn load_drone(&mut self) -> Option<Drone> {
        let config_id = self.config_id.clone()?;
        Some(self.loader.lock().unwrap().load_drone(&config_id))
    }

    pub fn load_replayer(&mut self, config_id: &str, replay_id: &str) -> Replayer {
        let drone = self.loader.lock().unwrap().load_drone(config_id);
        let sim_logs = self.loader.lock().unwrap().load_flight_log(replay_id);
        Replayer {
            drone,
            time: Duration::new(0, 0),
            time_accu: Duration::new(0, 0),
            time_steps: sim_logs,
            replay_index: 0,
            dt: Duration::from_nanos(5000),
        }
    }

    pub fn try_load_replay(&mut self) -> Option<Replayer> {
        if let (Some(config_id), Some(replay_id)) = (self.config_id.clone(), self.replay_id.clone())
        {
            Some(self.load_replayer(&config_id, &replay_id))
        } else {
            None
        }
    }

    pub fn insert_drone_rc(&mut self, controller_id: &str, controller: NonAdaptingDroneRc) {
        self.loader
            .lock()
            .unwrap()
            .insert_rc_controller(controller_id, controller);
    }

    pub fn load_drone_rc(&mut self, controller_id: &str) -> NonAdaptingDroneRc {
        self.loader
            .lock()
            .unwrap()
            .load_res_controller(controller_id)
    }

    pub fn insert_logs(&mut self, fl: FlightLog) {
        let mut logger = self.logger.lock().unwrap();
        for s in fl.steps {
            logger.log_time_stamp(s);
        }
        logger.flush();
    }
}
