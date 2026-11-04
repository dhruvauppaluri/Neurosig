# Formal verification

SystemVerilog Assertions for two RTL blocks, checked with SymbiYosys, Yosys (`read_slang`), and Bitwuzla. This directory is the place to read the contracts. Simulation testbenches stay under `rtl/tb/`.

| File | What it checks |
| --- | --- |
| `spike_count_decoder_formal.sv` | `rtl/spike_count_decoder.sv`: 32-channel Q8.8 dot product wrapped to 32 bits, and `prediction` for a non-negative score |
| `spike_count_decoder.sby` | k-induction (depth 1) and cover (depth 2) |
| `biquad_controller_formal.sv` | `rtl/biquad_controller.sv` plus `rtl/biquad_state.sv`: handshake, two-cycle latency, backpressure, reset, illegal-state recovery, per-channel memory isolation |
| `biquad_arithmetic_blackbox.sv` | Replaces `rtl/biquad_arithmetic.sv` in the controller proof. `new_y` is unconstrained |
| `biquad_controller.sby` | k-induction (depth 5) and cover (depth 16) |
| `run_formal.sh` | Runs both `.sby` files |

The controller proof does not check the Q2.30 multiply-accumulate. That comparison stays in `rtl/tb/biquad_arithmetic_tb.sv`. Neither proof is a timing, area, or power result, and neither covers the full filter-plus-detector pipeline.

`read_slang` keeps `(* anyseq *)` as an attribute and drives that net with a constant-X buffer. `biquad_controller.sby` deletes the buffer and attaches one `$anyseq` cell so the controller and the assertions sample the same `new_y`.

Run from the repository root, with the [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) on `PATH` or pointed to by `OSS_CAD_SUITE`:

```sh
OSS_CAD_SUITE=/path/to/oss-cad-suite bash formal/run_formal.sh
```

The latest pass is recorded in `reports/formal-verification.json`.
