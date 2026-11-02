# Neurosig visualization

Interactive research UI with a realistic human brain surface, synthetic spike generation, and Allen Visual Coding Neuropixels **sorted-spike** playback. This stack is isolated from RTL. The display is not a registered atlas, a biophysical network, or chip validation.

## Honesty

- The anatomical surface, cerebellum, brainstem, and tract meshes come from [Brain Project](https://github.com/itayinbarr/brainproject), based on Z-Anatomy / BodyParts3D. The asset is CC BY-SA 4.0; see [asset attribution](web/public/models/ATTRIBUTION.md). Neurosig's region volumes remain approximate ellipsoids. The GLB is visually fitted to them, not formally registered to an atlas.
- The cortex starts at 25% opacity and can be adjusted in the viewer. The 32-contact probe is the single illustrative visual-region array; the viewer does not imply recordings elsewhere.
- Whole-brain wiring and travel pulses are educational illustrations, not measured axonal conduction. Only synthetic source events animate along these paths.
- Synthetic activity is a toy stochastic model (region-dependent rates, bursts, weak correlation).
- Allen Visual Coding Neuropixels is a **mouse visual-cortex** dataset. Playback is mapped onto the occipital / visual region of that human-looking mesh only. Original CCF coordinates stay in event metadata and are never treated as human MNI.
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
cd viz/web && npm test && npm run build && cd ../..
```

Existing DSP tests remain:

```sh
.venv/bin/python -m unittest tests.test_quantize_adc tests.test_filter_iir tests.test_detect_spikes tests.test_fixed_point_model
```

## Modes

| Mode | What you see |
|---|---|
| `synthetic` | Randomized populations across approximate human-looking regions; illustrative travel paths and optional golden-model detections for visual-cortex sources |
| `neuropixels` | Fixture (or future NWB) sorted spikes in visual cortex only |
| `hybrid` | Both, with distinct event `kind` values |

Optional AllenSDK downloads: `.venv/bin/python -m viz.sim.download_allen` then convert a session to the JSON fixture schema. Full sessions stay under gitignored `data/neuropixels/`.

## Event schema

See [`schema/neural_event.v1.json`](schema/neural_event.v1.json). WebSocket batches: `WS /api/v1/events`. Control: `POST /api/v1/session/control`.

## Anatomy transform

The source GLB is in metres with +X anatomical left, +Y superior, and +Z anterior. `web/src/anatomyTransform.ts` maps it into the viewer's millimetre coordinates (+X right, +Y anterior, +Z superior): `[-840*x, 840*z + 7, 840*(y - 1.628)]`. The uniform 0.84 display scale aligns the asset visually with existing region and probe coordinates. This is not an anatomical registration; the mouse data remains a visual-region mapping only. The source GLB and a local Draco decoder are served from `web/public/` for offline playback.

On November 2, 2026, `npm test` passed four asset/transform/event checks, `npm run build` passed, and the 18 visualization Python tests passed. The built viewer was served locally on port 8766 and inspected at desktop and narrow widths in synthetic and recorded-spike modes. These checks cover the viewer, not neural detection accuracy or RTL.
