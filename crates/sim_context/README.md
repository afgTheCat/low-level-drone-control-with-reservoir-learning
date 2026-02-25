# sim_context

This crate provides a small “wiring layer” (`SimContext`) used to construct simulations and replays by combining:

- a `FlightController` implementation (Betaflight wrapper, reservoir controller, etc.)
- a `Logger` backend (file logger, rerun logger, etc.)
- a `Loader` implementation (file-based loader under `$HOME/.local/share/quad`, or defaults)

## Generating datasets

Datasets in this repo are just collections of recorded simulation episodes (`FlightLog`s) written by the file logger and later loaded by the file loader.

### `build_data_set` (surface-level)

The helper `build_data_set` in `crates/sim_context/src/input_gen.rs` generates a dataset by:

- creating a default `SimContext`
- selecting the Betaflight-based controller (`ControllerType::Betafligt`)
- switching to the file-based loader (`LoaderType::File`)
- generating random control input sequences (currently Brownian noise on throttle/yaw/pitch/roll)
- running multiple simulation “episodes” and logging each episode via `LoggerType::File`

Each episode is simulated with a fixed timestep of `1ms` for the requested duration and is stored as JSON.

### Where the files go

`LoggerType::File` writes logs under:

- `$HOME/.local/share/quad/replays/<dataset_id>/training_<n>`
- `$HOME/.local/share/quad/replays/<dataset_id>/testing_<n>`

The file loader’s `load_data_set(dataset_id)` expects exactly this layout and uses the `training_` / `testing_` prefixes to split train/test episodes (see `crates/loaders/src/file_loader/mod.rs`).

### How to run it

Right now dataset generation is exposed as a unit test:

- `cargo test -p sim_context input_gen::test::build_50 -- --nocapture`

This will generate a few example datasets (see the test body in `crates/sim_context/src/input_gen.rs`).
