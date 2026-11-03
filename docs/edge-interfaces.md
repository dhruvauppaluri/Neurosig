# Edge stack interface freeze

Shared assumptions for localization, the software ASIC link, decoding, and visualization.

- Seed `20261103`, sample rate 32 kHz, 32 channels.
- Snippets: 1.0 ms before and 1.5 ms after the event sample (`edge.SNIPPET_PRE_MS` / `SNIPPET_POST_MS`).
- Scientific frame is `probe_um`. Display frame is `stylized_human` millimetres via `edge.coords`.
- Packet magic `NSG1`, little-endian, CRC32, `schema_version=1`. Layouts in `edge/link/schema.py`.
- Session localization JSON: `{available, method, xyz_mm, error_um, confidence, unverified, reason}`.
- Event kind `localization_estimate` already exists; do not add competing kinds.
- `viz` may import `edge`. `edge` must not import `viz.web`.
- Do not modify `rtl/` or golden-model arithmetic in `scripts/filter_iir.py`, `scripts/detect_spikes.py`, `scripts/fixed_point_model.py`, `scripts/quantize_adc.py`.
- Reuse `scripts.synthetic_decoder.train` for linear decoding. Preserve `rtl/spike_count_decoder.sv` as the count-only RTL baseline.
- Labels are synthetic or recorded stimulus conditions, not visual perceptions. Neuropixels CCF is not a human neuron coordinate.
