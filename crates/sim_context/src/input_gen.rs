use crate::SimContext;
use flight_controller::Channels;
use rand::{distributions::Bernoulli, prelude::Distribution, thread_rng};
use std::time::Duration;

fn generate_brownian(milisecs: u128) -> Vec<f64> {
    let bernoulli = Bernoulli::new(0.5).unwrap();
    let mut rng = thread_rng();
    let axis = (0..milisecs).fold((0., 0., vec![]), |acc, _| {
        let (mut pos, mut vel, mut all_pos) = acc;
        vel += if bernoulli.sample(&mut rng) {
            0.0001
        } else {
            -0.0001
        };
        pos += vel;
        if !(-1. ..=1.).contains(&pos) {
            // why do we need this?
            pos = f64::clamp(pos, -1000., 1000.);
            vel = 0.;
        }
        pos = pos.clamp(-1., 1.);
        all_pos.push(pos);
        (pos, vel, all_pos)
    });
    axis.2
}

pub enum InputGenerationMethod {
    Uniform(f64),
    Brownian,
}

impl InputGenerationMethod {
    fn to_values(&self, milisecs: u128) -> Vec<f64> {
        match self {
            Self::Uniform(val) => vec![*val; milisecs as usize],
            Self::Brownian => generate_brownian(milisecs),
        }
    }
}

pub struct InputGenerator {
    throttle: InputGenerationMethod,
    yaw: InputGenerationMethod,
    pitch: InputGenerationMethod,
    roll: InputGenerationMethod,
}

impl Default for InputGenerator {
    fn default() -> Self {
        InputGenerator {
            throttle: InputGenerationMethod::Uniform(-1.),
            yaw: InputGenerationMethod::Uniform(0.),
            pitch: InputGenerationMethod::Uniform(0.),
            roll: InputGenerationMethod::Uniform(0.),
        }
    }
}

impl InputGenerator {
    fn set_throttle(self, throttle: InputGenerationMethod) -> Self {
        Self { throttle, ..self }
    }

    fn set_yaw(self, yaw: InputGenerationMethod) -> Self {
        Self { yaw, ..self }
    }

    fn set_pitch(self, pitch: InputGenerationMethod) -> Self {
        Self { pitch, ..self }
    }

    fn set_roll(self, roll: InputGenerationMethod) -> Self {
        Self { roll, ..self }
    }

    fn generate(&self, duration: Duration) -> Vec<Channels> {
        let milisecs = duration.as_millis();
        let throttle = self.throttle.to_values(milisecs);
        let yaw = self.yaw.to_values(milisecs);
        let pitch = self.pitch.to_values(milisecs);
        let roll = self.roll.to_values(milisecs);

        let mut channels = Vec::with_capacity(milisecs as usize);
        for ms in 0..milisecs {
            let idx = ms as usize;
            channels.push(Channels {
                throttle: throttle[idx],
                yaw: yaw[idx],
                pitch: pitch[idx],
                roll: roll[idx],
            });
        }
        channels
    }
}

pub fn build_data_set(
    data_set_id: String,
    training_duration: Duration,
    training_size: usize,
    test_size: usize,
) {
    let mut context = SimContext::default();
    context.set_controller(crate::ControllerType::Betafligt);
    let input_generator = InputGenerator::default()
        .set_throttle(InputGenerationMethod::Brownian)
        .set_yaw(InputGenerationMethod::Brownian)
        .set_pitch(InputGenerationMethod::Brownian)
        .set_roll(InputGenerationMethod::Brownian);
    context.set_loader(&crate::LoaderType::File);
    let training_inputs = (0..training_size)
        .map(|_| input_generator.generate(training_duration))
        .collect::<Vec<_>>();
    let test_inputs = (0..test_size)
        .map(|_| input_generator.generate(training_duration))
        .collect::<Vec<_>>();

    // TODO: do this on multiple cores
    for (ep, inputs) in training_inputs.into_iter().enumerate() {
        let simulation_id = format!("{data_set_id}/training_{ep}");
        context.set_logger(crate::LoggerType::File(simulation_id));
        let mut simulation = context.try_load_simulator().unwrap();
        simulation.init();
        for input in inputs {
            simulation.simulate_delta(Duration::from_millis(1), input);
        }
    }

    for (ep, inputs) in test_inputs.into_iter().enumerate() {
        let simulation_id = format!("{data_set_id}/testing_{ep}");
        context.set_logger(crate::LoggerType::File(simulation_id));
        let mut simulation = context.try_load_simulator().unwrap();
        simulation.init();
        for input in inputs {
            simulation.simulate_delta(Duration::from_millis(1), input);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::input_gen::build_data_set;
    use std::time::Duration;

    #[test]
    fn build_50() {
        build_data_set("5_len".into(), Duration::from_secs(5), 5, 5);
        build_data_set("10_len".into(), Duration::from_secs(5), 10, 10);
        build_data_set("25_len".into(), Duration::from_secs(5), 25, 25);
        build_data_set("50_len".into(), Duration::from_secs(5), 50, 50);
    }
}
