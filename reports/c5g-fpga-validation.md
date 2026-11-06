# C5G FPGA validation — 2026-11-03 to 2026-11-05

Board target: Terasic C5G, `5CGXFC5C6F27C7N`. Quartus Prime Lite 25.1std
build 1129 and a USB-Blaster were used on the Windows host. Quartus
Programmer configured the C5G; the physical results appear below.

## Reference provenance

The seed-42 MEArec workflow was regenerated in WSL with pinned Python 3.12.15
and MEArec 1.11.0. The hardware reference uses
`scripts/unfiltered_mearec.py`, followed by `scripts/quantize_adc.py`,
`scripts/filter_iir.py`, and `scripts/fixed_point_model.py`.

| Artifact | SHA-256 |
| --- | --- |
| `data/unfiltered/recording.h5` | `94299c4a74bbf4ce2aa2e528ba28e80122cc6d051f8e55632ad3838d4f215055` |
| `data/quantized/adc_codes_i16le.bin` | `73b1ea0c55a390bd55f7fc3ff388eaad47908836b3576b53d0ce5693fd726c5e` |
| `data/fixed_point/threshold_codes.npy` after clean regeneration | `3f33dae937e57d97ecc21963f08725a618348e41cc7f96b30dee088396483d41` |

The regenerated Python event triples equal all nine entries in
`reports/rtl-pipeline-validation.json`. The separately regenerated filtered
smoke recording produces a different ADC vector and must not be compared to
that reference.

On November 5, the stored `data/filtered/filtered_uv.npy` was found to be
stale relative to the regenerated ADC bytes. Re-running `scripts/filter_iir.py`
and `scripts/fixed_point_model.py` from those bytes produced the threshold hash
above and retained the same nine exact event triples. The earlier physical
run used its original threshold hex file (SHA-256
`c73f629c3a1c02aaa3e0c262b5198700f195331c6d20e5d3a91b6faa1174202e`),
preserved in ignored `data/fpga/c5g-original-thresholds.hex`. The clean
thresholds are also reproduced by `fpga/prepare_raw.py` from the seeded
microvolt array; its output ADC bytes match the original byte for byte.

## Completed checks

- `python -m unittest discover -s tests`: 21 passed.
- `python -m unittest discover -s edge/tests`: 55 passed.
- Icarus directed detector and event merger tests: passed.
- Icarus C5G mailbox test: CRC rejection, sequence replay prevention, channel
  order rejection, 24-bit event serialization, and event FIFO overflow passed.
- Verilator 5.050 pipeline reference: 1,024,000 samples and nine exact events.
- Verilator 5.050 C5G mailbox wrapper: 1,024,000 accepted samples and nine
  exact buffered events; 2,044,000 simulated ready/valid stall cycles under
  the block-loading test schedule.
- Verilator 5.050 dense seeded pipeline regressions at 1, 4, and 32 channels:
  5 exact merged events at each channel count after the fixed-slot event merger
  refactor. The direct merger and integrated reset tests also passed.
- Quartus analysis and elaboration: passed with five warnings on the C5G
  device.
- The 32-slot combinational merger build completed Quartus synthesis, fit,
  timing, and `.sof` generation. It used 20,065 / 29,080 ALMs (69%), 19,012
  registers, 57,344 block memory bits, and 30 DSP blocks. It **failed** the
  20 ns clock constraint: worst setup slack −332.118 ns and Fmax 2.84 MHz.
  The critical path crossed 338 logic levels inside event selection. This
  bitstream was not programmed.
- The C5G wrapper now uses a clocked 32-slot event merger. Its full-trace
  Verilator 5.050 test accepted 1,024,000 samples and produced the same nine
  exact events with peak merger occupancy three. The original combinational
  merger and default pipeline regression also passed after the selection
  parameter was added. Quartus fitted this version at 15,880 ALMs (55%), but
  setup slack was still −118.091 ns. Its worst path included the generic
  32-channel complete-frame minimum. The mailbox validates strict frame-major
  input, so the current C5G variant derives completed frames from the final
  channel counter; full-trace Verilator still matches all nine events. Quartus
  fitted that variant at 14,658 ALMs (50%) with setup slack −9.561 ns. The
  worst path then ran through a high-pass biquad state read and arithmetic.
  Static per-channel register enables did not close that path (−11.306 ns
  setup slack). The current C5G variant schedules state read, multiplication,
  accumulation, and saturation across separate clocks. Its 32,768-output
  Python-to-RTL filter comparison passes exactly with backpressure, and its
  full 1,024,000-sample Verilator wrapper comparison still produces the nine
  exact events. Its fitted and measured results are recorded below.

## Physical acceptance — 2026-11-05

Quartus Prime Lite 25.1std generated a `.sof` for the pipelined C5G variant.
The fitter used 14,980 / 29,080 ALMs (52%), 19,020 registers, 59,296 block
memory bits (10 RAM blocks), and 30 / 150 DSP blocks. At the slow 85°C corner,
the 50 MHz clock had +3.906 ns worst setup slack and 62.13 MHz reported Fmax.
The USB-Blaster Programmer configured one C5G device successfully with `.sof`
checksum `0x02B2E7D6`; System Console read the `NSG4` mailbox magic.

Zero, impulse, and patterned 4,096-sample runs completed with zero events and
zero error flags under the seeded thresholds. The complete seeded recording
was run twice. The measured fresh run accepted 1,024,000 samples, captured
all nine exact `(sample, channel, signed 24-bit amplitude)` triples in order,
and matched both the regenerated Python report and a fresh Verilator output
file. There were zero missing or extra events, zero sticky errors, zero event
status flags, and maximum observed event FIFO occupancy one. The original
24-bit records were retained; all nine fit the NSG1 signed-16 amplitude range,
and `edge.pipeline.ingest` accepted nine converted EVENT frames.
Nine 49×32 host ADC waveform windows were also retrieved at the hardware
timestamps; with the MEArec channel-position array, the edge pipeline
ingested and localized all nine. These snippets and positions came from the
host recording, not from the FPGA.

The fresh System Console playback took **746.935 s**, or **1,370.94 samples/s**
end to end. This includes host preparation, System Console startup, 1,000
block transfers, status polling, and final flush. It is about 747 times below
the 1.024 million samples/s transfer target; the result is a valid offline
playback milestone, not a real-time JTAG link. The simulator measured
10,243,216 cycles for the wrapper test, including its simulated bus writes,
and 4,087,000 ready/valid stall cycles. Those are simulation counts, not
hardware cycle counters. A counter-equipped revision is being compiled for
direct on-chip processing measurement.

The frozen real-data CAR/4σ Python detector is not an equivalent RTL reference
because the hardware lacks median CAR. No real-data board comparison is
claimed yet.

## Counter-equipped final build — 2026-11-05

Quartus full compilation completed with zero errors. Programmer loaded the
counter-equipped `.sof` (checksum `0x02B69090`) over `USB-Blaster [USB-0]`.
The fitted device uses 15,097 / 29,080 ALMs (52%), 19,084 registers,
59,808 block-memory bits in 10 RAM blocks, and 30 / 150 DSP blocks. The
slow 85°C timing model reports +5.050 ns worst setup slack for the 20 ns
clock and 67.27 MHz Fmax. The fitted one-pin design uses the board's
documented `CLOCK_50_B5B` at `PIN_R20`.

A 4,096-sample zero run on this revision had no events or errors. The fresh
1,024,000-sample board run matched all nine Python and Verilator triples in
order with zero missing or extra records, zero event status flags, zero
sticky errors, and event FIFO high-water occupancy one. Its FPGA counters
reported 6,135,000 feed clocks, 2,048,000 CRC/order-check clocks, and nine
emitted events. Feed time was 5.991211 clocks/sample. At 50 MHz this is an
8.345 million samples/s **on-chip feed capacity** for this trace, above the
1.024 million samples/s processing target. It excludes the JTAG host path.
The measured end-to-end host run took 809.964 s for 1,024,000 samples,
or 1,264.25 samples/s, including startup, transfer, polling, and flush.
This is about 810 times slower than the 1.024 million samples/s real-time
transfer target.
The input CRC and channel-order checks completed with no errors. Output
records are retained in ignored `data/fpga/c5g-counters-events.json` and `.csv`.

`quartus_pow neurosig_c5g --use_vectorless_estimation=on` estimated 373.56 mW
total thermal power (140.62 mW core dynamic, 227.42 mW core static,
5.52 mW I/O). Quartus rated this estimate **low confidence** because no
measured switching trace was provided. It is neither a measured FPGA power
value nor total board power.

The clean-calibration replay used regenerated float and fixed-point files.
Its prepared ADC bytes and Python event triples matched the seeded reference;
its threshold hex SHA-256 was
`c0b5415c3dd9e7c2bee6ad2f2d51a5af70521c7daf4b7552134ad1167c776fc0`.
The same programmed board then processed all 1,024,000 samples and again
matched all nine Python and Verilator triples, with zero errors and FIFO
high-water one. This fresh run took 760.947 s end to end (1,345.69
samples/s). It is recorded in ignored
`data/fpga/c5g-clean-calibration-events.json` and `.csv`.

## Public raw AP window

`fpga/fetch_ibl_window.py` fetched a public IBL AP window from insertion
`4836a465-c691-4852-a0b1-dcd2b1ce38a1`, starting at 100 s for 1 s. It
selected 32 VISp channels with native 30 kHz calibrated AP voltages. The
voltage span was −278.91 to +189.84 µV. `fpga/prepare_raw.py` resampled to
32 kHz, quantized at an explicit ±300 µV range with zero saturation, and
calibrated the no-CAR 6σ thresholds from the quantized input. Its bit-accurate
Python reference produced one event. The board comparison is pending.
The ignored raw `.npy`, prepared ADC, and threshold hex SHA-256 values are
`23d3b6ba93d5a635b2228007eb6d1aec284bb0f033670e99f82a4d8790d7d03a`,
`63bf0340f70c4a2ea4e1128ec51ffe2400b858a4033ad7bd59c68730d531413a`,
and `b26510d8fcfb01b31537bfd7fb6f8f60ebf90edd4231ac9f7a40c18a3f61f4cb`
respectively. The source and channel selection are in ignored
`data/fpga/ibl-window/source.json`.
