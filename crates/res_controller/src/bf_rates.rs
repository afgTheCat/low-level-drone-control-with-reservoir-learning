use flight_controller::Channels;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Rates {
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
}

pub struct BetaflightRates {
    pub max_rate: f64,    // deg/s or rad/s
    pub center_rate: f64, // The slope at the center
    pub expo: f64,        // Curvature (0.0 = quadratic)
}

impl BetaflightRates {
    /// Implements the "Actual Rates" system (Betaflight 4.3+)
    pub fn actual_rate(&self, stick: f64) -> f64 {
        let x = stick.abs().clamp(0.0, 1.0);
        let s = stick.signum();

        // The power is usually (1 + expo) or (2 + expo) depending on BF version
        // For your data, a total power of 2 (linear + quadratic) fits best.
        let power = 1.0 + self.expo;

        let rate = x * self.center_rate + (self.max_rate - self.center_rate) * x.powf(1.0 + power);

        s * rate
    }
}

impl Default for BetaflightRates {
    fn default() -> Self {
        BetaflightRates {
            max_rate: 11.682,
            center_rate: 1.19,
            expo: 0.,
        }
    }
}

pub fn stick_input_to_angular_velocity(stick_input: f64) -> f64 {
    BetaflightRates::default().actual_rate(stick_input)
}

pub fn stick_inputs_to_targets(stick_inputs: &Channels) -> Rates {
    let yaw = stick_input_to_angular_velocity(stick_inputs.yaw);
    let pitch = -stick_input_to_angular_velocity(stick_inputs.pitch);
    let roll = -stick_input_to_angular_velocity(stick_inputs.roll);
    Rates { yaw, pitch, roll }
}
