# Low level drone control with reservoir learning

This repository contains the code supporting my master thesis: *Learning Low-Level Drone Control with
Reservoir Computing*.

## Quickstart

- Run the visualizer UI:
  - `cargo run -p visualizer`
- Reproduce the thesis experiments via the training/evaluation test suite:
  - `cargo test -p res_controller_training -- --nocapture`

The visualizer (when using the file-based loader) discovers saved reservoir controllers under `$HOME/.local/share/quad/reservoirs` (see `crates/loaders/src/file_loader/mod.rs`).

## Repository structure

This is a Rust workspace (`Cargo.toml`) with most code living under `crates/`.

### Top-level

- `crates/`: Rust crates (simulation, controllers, training, visualization).
- `libvirtual_betaflight.so`: shared library used by `crates/bf_controller` (loaded at runtime).
- `eeprom.bin`, `eeprom7in.bin`: EEPROM/config blobs used to initialize the Betaflight-based controller.
- `pre_trained_controllers/`: saved controller parameters/checkpoints used by loaders/training code.

### Crates (high level)

- `crates/visualizer`: Bevy UI app to run simulations and view replays.
- `crates/simulator`: simulation loop + replay runner (glues drone + controller + logger).
- `crates/drone`: drone dynamics/model and related utilities.
- `crates/flight_controller`: common controller traits and shared types (channels, updates, motor outputs).
- `crates/bf_controller`: `dlmopen`-based wrapper around the virtual Betaflight shared library.
- `crates/sim_context`: “app context” wiring for choosing loader/logger/controller and constructing sims/replays.
- `crates/loggers`: logging backends and `FlightLog` types (file / rerun / in-memory).
- `crates/loaders`: load drones, controllers, replays, and datasets (default + file-based loaders).
- `crates/res`: reservoir-related models (e.g. ESN, Izhikevich).
- `crates/ridge`: ridge regression utilities.
- `crates/res_controller`: reservoir-based flight controllers and input mappings.
- `crates/res_controller_training`: training/evaluation utilities for reservoir controllers (has its own README/tests).
- `crates/macros`: small proc-macro helpers used across the workspace.

