# res_controller_training

Training and evaluation utilities used to reproduce experiments for *Low level drone control with reservoir learning*.

Most “reproducible results” in this crate are expressed as `cargo test` integration tests.

## Prerequisites (data on disk)

Several tests expect datasets and/or trained controllers to be available via the file-based loader under:

- `$HOME/.local/share/quad/replays/` (datasets / recorded episodes)
- `$HOME/.local/share/quad/reservoirs/` (saved reservoir controllers)

See `crates/sim_context/README.md` for how datasets are generated in this repo.

## Key tests

### `esn_parameter_sweep_test`

Location: `crates/res_controller_training/tests/results.rs`

What it does (high level):

- sweeps ESN hyperparameters (buffer size × PCA dimension)
- trains/evaluates each configuration on a dataset (currently `5_len_inc`)
- appends per-episode MSE results to `crates/res_controller_training/combined5.csv`

Run:

- `cargo test -p res_controller_training --test results esn_parameter_sweep_test -- --nocapture`

### `evaluate_esn_stabilization`

Location: `crates/res_controller_training/tests/results.rs`

What it does (high level):

- loads datasets (`5_len`, `10_len`, `25_len`, `50_len`) via the file loader
- trains an ESN controller on each dataset (after downsampling)
- stores the trained controllers back to `$HOME/.local/share/quad/reservoirs/` (via `SimContext::insert_drone_rc`)
- evaluates closed-loop angular-rate stabilization in the simulator and prints a success rate

Run:

- `cargo test -p res_controller_training --test results evaluate_esn_stabilization -- --nocapture`

