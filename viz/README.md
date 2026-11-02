# Neurosig visualization

Interactive research UI for stylized whole-brain activity, synthetic spike generation, and Allen Visual Coding Neuropixels **sorted-spike** playback. This stack is isolated from RTL. It is not an anatomically exact brain, not a biophysical network, and not chip validation.

## Honesty

- The 3D brain is a sci-fi / educational mesh with named ellipsoidal regions.
- Synthetic activity is a toy stochastic model (region-dependent rates, bursts, weak correlation).
- Allen Visual Coding Neuropixels is a **mouse visual-cortex** dataset. Playback is confined to a homologous stylized visual-cortex volume. Original CCF coordinates stay in event metadata and are never treated as human MNI.
- Mode B plays already-sorted spike times. Sorted timestamps are never fed to the ADC.
- Golden-model detections are offline software windows using the existing float ADC / IIR / detector APIs. They are not RTL results.
- Localization is a labeled placeholder: **Localization not implemented**.

## Setup

Use the project virtualenv (do **not** `uv pip sync` the MEArec lock after this; that would drop visualization extras):

```sh
.tools/bin/uv pip install --python .venv/bin/python -r viz/requirements.txt
cd viz/web && npm install && npm run build && cd ../..
.venv/bin/python -m viz.sim.server
```

Open http://127.0.0.1:8765/

Frontend dev server (proxies `/api` to port 8765):

```sh
.venv/bin/python -m viz.sim.server
cd viz/web && npm run dev
```

## Tests

```sh
.venv/bin/python -m unittest viz.tests.test_schema viz.tests.test_synthetic viz.tests.test_neuropixels viz.tests.test_golden_bridge viz.tests.test_server
```

Existing DSP tests remain:

```sh
.venv/bin/python -m unittest tests.test_quantize_adc tests.test_filter_iir tests.test_detect_spikes tests.test_fixed_point_model
```

## Modes

| Mode | What you see |
|---|---|
| `synthetic` | Randomized populations across stylized regions; optional electrode rendering + golden-model detections for visual-cortex sources |
| `neuropixels` | Fixture (or future NWB) sorted spikes in visual cortex only |
| `hybrid` | Both, with distinct event `kind` values |

Optional AllenSDK downloads: `.venv/bin/python -m viz.sim.download_allen` then convert a session to the JSON fixture schema. Full sessions stay under gitignored `data/neuropixels/`.

## Event schema

See [`schema/neural_event.v1.json`](schema/neural_event.v1.json). WebSocket batches: `WS /api/v1/events`. Control: `POST /api/v1/session/control`.
