# C5G JTAG playback

Target: Terasic Cyclone V GX Starter Kit (C5G), `5CGXFC5C6F27C7N`.
The only external logic pin is the documented 50 MHz `CLOCK_50_B5B` at
`PIN_R20` (3.3 V). USB-Blaster uses dedicated JTAG pins. USB-UART is unused.

## Build and program

Quartus Prime Lite 25.1 is installed at `C:\altera_lite\25.1std` on the
validation PC. Quartus cannot create its database in this OneDrive folder;
copy `rtl/` and `fpga/` together to an ordinary local folder before building.
Run from that copy's `fpga/` directory:

```powershell
& 'C:\altera_lite\25.1std\quartus\sopc_builder\bin\qsys-generate.exe' neurosig_jtag.qsys --synthesis=VERILOG --family='Cyclone V' --part=5CGXFC5C6F27C7
& 'C:\altera_lite\25.1std\quartus\bin64\quartus_sh.exe' --flow compile neurosig_c5g
& 'C:\altera_lite\25.1std\quartus\bin64\quartus_pgm.exe' -c 'USB-Blaster [USB-0]' -m jtag -o 'p;neurosig_c5g.sof'
```

Generated `neurosig_jtag/`, `db/`, and `*.sof` are ignored. A `.sof`
program is volatile. Check the timing report against the 20 ns constraint.

## Mailbox protocol

Addresses are byte addresses; transfers are 32-bit little-endian. The 50 MHz
clock runs both the mailbox and DSP. Intel JTAG IP handles its own JTAG crossing.

| Address | Access | Meaning |
| --- | --- | --- |
| `0x00` | R | magic `0x4e534734` (`NSG4`) |
| `0x04` | R | protocol version (currently `0x00010001`) |
| `0x08` | R | status: bit 0 busy, bit 1 event pending, bit 2 idle |
| `0x0c` | R | sticky errors: bit 0 event FIFO full, 1 merge overflow, 2 busy write, 3 bad command/sequence, 4 channel order, 5 CRC/order rejection |
| `0x10` | R/W | session ID |
| `0x14` | R/W | block sequence, starting at 0 |
| `0x18` | R/W | block length, 1–1024 words |
| `0x1c` | R/W | IEEE CRC32 of the little-endian input words |
| `0x20` | R | last acknowledged sequence (`0xffffffff` initially) |
| `0x24` | R | total accepted samples |
| `0x28` | R | output event FIFO occupancy |
| `0x2c` | R | head event sample index |
| `0x30` | R | head event: signed 24-bit amplitude, channel bits 28:24, valid bit 29 |
| `0x34` | R | head event sequence |
| `0x38` | R | running block CRC32 result |
| `0x3c` | W | write 1 to pop one event |
| `0x40` | W | `REST`=`0x52455354`, `COMT`=`0x434f4d54`, `FLSH`=`0x464c5348` |
| `0x44` | R | FPGA clocks spent feeding the most recent recording |
| `0x48` | R | FPGA clocks spent validating committed blocks |
| `0x4c` | R | event FIFO high-water occupancy |
| `0x50` | R | total emitted event records |
| `0x100`–`0x17c` | R/W | 32 signed 24-bit thresholds |
| `0x1000`–`0x1ffc` | W | 1024 input words: signed sample bits 15:0, channel bits 20:16 |

Write block RAM and metadata, then `COMT`. The RAM uses a synchronous read
port inside the FPGA; its contents are not exposed to the host. CRC and strict frame-major
channel order are checked before processing. A repeated acknowledged sequence
does not replay data. Poll idle and acknowledgment, then drain events. The
host saves each event before popping it so resumed runs can deduplicate by
event sequence. Send `FLSH` after the final block; the wrapper allows 32
filter-drain clocks and 4096 flush clocks for the clocked merge scan. `REST`
starts a new finite recording.

This format preserves the pipeline's signed 24-bit event amplitude. NSG1's
current EVENT packet has only a signed 16-bit amplitude; conversion must reject
out-of-range values. The hardware does not transmit waveform snippets; obtain
them from the host recording by timestamp.

The C5G instance has 32 merge-candidate slots and scans them over multiple
clocks to avoid a long selection path. The generic RTL pipeline keeps its
64-slot combinational reference merge. The seeded full trace reached three
occupied slots. Merge overflow is sticky status bit 1 and invalidates a run.
The C5G filter schedules the same five products, accumulation, and saturation
over separate clocks. Its 32,768-output Python comparison passes exactly;
the board clock and on-chip cycles per sample are reported separately.

## Reproduce and run

```sh
.venv/bin/python scripts/smoke_mearec.py
.venv/bin/python scripts/unfiltered_mearec.py
.venv/bin/python scripts/quantize_adc.py
.venv/bin/python scripts/filter_iir.py
.venv/bin/python scripts/fixed_point_model.py
.venv/bin/python rtl/tb/generate_pipeline_reference.py --threshold-output data/fixed_point/thresholds.hex
python3 fpga/run_c5g.py --adc data/quantized/adc_codes_i16le.bin --thresholds data/fixed_point/thresholds.hex --output data/fpga/c5g-events.json
```

`run_c5g.py` works with Windows Python or WSL Python and invokes Windows
System Console. It checks event sequences and exact event triples against
`reports/rtl-pipeline-validation.json`; pass `--verilator-events` with the
full-pipeline observed triples for a second independent file comparison. Use
`--resume` with the same session
after an interruption, without resetting or reprogramming the FPGA. The
first acceptance target is the nine-event seeded recording. Report measured
JTAG playback separately from on-chip processing capacity.

After compilation, `python3 fpga/performance_report.py --build-dir
<local-build>/fpga --output data/fpga/performance.json` collects the
utilization and timing lines actually present in Quartus reports. Add
`--hardware-json data/fpga/c5g-events.json` after physical playback to include
measured host transfer time. Missing report fields stay empty.

For small board checks, run `python3 fpga/make_vectors.py --kind zeros
--frames 128 --output data/fpga/zeros.bin` (or `impulse` / `pattern`), then use
`run_c5g.py --adc data/fpga/zeros.bin --skip-compare` with the same threshold
file and a fresh output path. Each invocation resets the FPGA session.

After a verified run, `.venv/bin/python fpga/export_edge_events.py
data/fpga/c5g-events.json data/fpga/c5g-events.nsg1` exports EVENT frames for
the existing host edge stack and runs them through `edge.pipeline.ingest`,
writing a `.edge.json` receipt. The converter range-checks the 16-bit NSG1
amplitude field; the original 24-bit codes remain in the hardware JSON.

`retrieve_snippets.py` reads event-centered windows from the **host ADC file**.
For example, pass a `[32,3]` electrode-position `.npy` file with
`--positions-npy` and the recording's `--lsb-uv` to run the existing edge
localizer on those snippets. The `.npz` keeps signed ADC codes, and the
sidecar JSON records the source and window bounds. This is host waveform
retrieval, not an FPGA snippet transmitter.

Real Neuropixels AP recordings need 32 kHz adaptation, 16-bit quantization,
and matching offline thresholds. The frozen Python CAR/4σ detector does not
match the current no-CAR RTL.

`prepare_raw.py` accepts a `[samples, channels]` `.npy` array of microvolt
values. Select exactly 32 columns with `--channels-file` for wider recordings.
Specify the input sample rate and the ADC half-range explicitly. For example:

```sh
.venv/bin/python fpga/prepare_raw.py data/raw/window_uv.npy data/fpga/raw-window \
  --sample-rate-hz 30000 --full-scale-uv 135 --channels-file data/raw/channels.json
python3 fpga/run_c5g.py --adc data/fpga/raw-window/adc_i16le.bin \
  --thresholds data/fpga/raw-window/thresholds.hex \
  --expected data/fpga/raw-window/expected.json \
  --output data/fpga/raw-window/hardware.json
```

Preparation resamples to 32 kHz when needed, rejects saturated ADC values by
default, calibrates offline 6σ thresholds on the reconstructed ADC voltages,
and writes the bit-accurate no-CAR fixed-point event reference. The host run
compares exact event triples with that reference. A matching raw-data result
applies only to the selected channels, window, ADC range, and resampling
choice; it does not validate the frozen median-CAR real detector.

For a public IBL AP window, first run `bash scripts/setup_ibl.sh`, then use
the optional locked IBL environment:

```sh
data/ibl-venv/bin/python fpga/fetch_ibl_window.py \
  --start-s 100 --duration-s 1 --output-dir data/fpga/ibl-window
.venv/bin/python fpga/prepare_raw.py \
  data/fpga/ibl-window/window_uv.npy data/fpga/ibl-prepared \
  --sample-rate-hz 30000 --full-scale-uv 300
```

Read the actual `sample_rate_hz` in `ibl-window/source.json` and pass that
value to `prepare_raw.py`; 30 kHz above is an example. The fetcher saves the
selected 32-channel voltage matrix, raw channel indices, and local probe
positions. The input is public calibrated AP voltage, not sorted event counts.

Quartus Power Analyzer can provide a vectorless estimate with
`quartus_pow neurosig_c5g --use_vectorless_estimation=on`. Its confidence
depends on supplied activity; the recorded vectorless result is low
confidence and does not measure board power.
