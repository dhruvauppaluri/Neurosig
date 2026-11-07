# Neurosig visualization

Isolated from RTL. Stylized human-looking anatomy, toy synthetic spikes, and Allen Visual Coding **sorted-spike** playback mapped onto the visual region of that mesh.

The surface is [Brain Project](https://github.com/itayinbarr/brainproject) (Z-Anatomy / BodyParts3D, CC BY-SA 4.0; [attribution](web/public/models/ATTRIBUTION.md)), visually fitted to ellipsoid region volumes — not a registered atlas. Wiring and travel pulses are illustrations. Estimated locations use a 1/r amplitude model on electrode snippets. Mouse CCF stays in metadata and is not treated as human MNI. Mode B never feeds sorted times into the ADC. Golden-model detections in the viewer are offline Python, not FPGA/RTL.

Scientific error is micrometres (`error_um`); viewer markers are stylized millimetres via `edge.coords`. Truth, electrodes, and estimates use different markers.

## Setup

Use the project venv. Do not `uv pip sync` the MEArec lock after this; that drops visualization extras.

```sh
.tools/bin/uv pip install --python .venv/bin/python -r viz/requirements.txt
cd viz/web && npm install && npm run build && cd ../..
.venv/bin/python -m viz.sim.server
```

Open http://127.0.0.1:8765/

Frontend dev server (proxies `/api` to 8765):

```sh
.venv/bin/python -m viz.sim.server
cd viz/web && npm run dev
```

## Tests

```sh
.venv/bin/python -m unittest viz.tests.test_schema viz.tests.test_synthetic viz.tests.test_neuropixels viz.tests.test_golden_bridge viz.tests.test_server
.venv/bin/python -m unittest discover -s edge/tests
cd viz/web && npm test && npm run build && cd ../..
```

## Modes

| Mode | What you see |
|---|---|
| `synthetic` | Toy populations on approximate regions; optional golden-model detections for visual-cortex sources |
| `neuropixels` | Fixture (or NWB) sorted spikes in visual cortex only |
| `hybrid` | Both, distinct event `kind` values |

Optional AllenSDK: `.venv/bin/python -m viz.sim.download_allen`. Full sessions stay under gitignored `data/neuropixels/`.

## Event schema

[`schema/neural_event.v1.json`](schema/neural_event.v1.json). Batches: `WS /api/v1/events`. Control: `POST /api/v1/session/control` (includes `localization_method`). Snippets: `GET /api/v1/snippet/{event_id}`. Offline aggregates: `GET /api/v1/research`.

## Anatomy transform

Source GLB is metres, +X left, +Y superior, +Z anterior. `web/src/anatomyTransform.ts` maps to viewer millimetres (+X right, +Y anterior, +Z superior): `[-840*x, 840*z + 7, 840*(y - 1.628)]`. Scale 0.84 is a visual fit, not registration. GLB and a local Draco decoder are under `web/public/` for offline playback.
