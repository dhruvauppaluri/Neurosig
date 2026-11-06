"""Informational: integer v2 detector with thresholds calibrated on a DIFFERENT held-out window
of the same session (next window, cyclic) versus self-calibration. No tuning; reports only."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.phase5.benchmark import RADIUS_UM, load_split
from scripts.phase5.compare_float_fixed import spec_from_float
from scripts.phase5.detectors import P5Config
from scripts.phase5.fixed_detector import adc_quantize, run_fixed
from scripts.phase5.metrics import pooled, window_metrics

v2 = P5Config(highpass_hz=200.0, lowpass_hz=4500.0)
spec = spec_from_float(v2)
items = load_split("heldout")
by = {}
for i, (_, s) in enumerate(items):
    by.setdefault(s["session"], []).append(i)
runs = {i: run_fixed(adc_quantize(w["voltages_uv"], spec), spec) for i, (w, _) in enumerate(items)}
out = {"self": [], "other_window": []}
for idx in by.values():
    for k, i in enumerate(idx):
        w, _ = items[i]
        cal = runs[idx[(k + 1) % len(idx)]]["calibration"]
        for mode, r in (("self", runs[i]), ("other_window", run_fixed(adc_quantize(w["voltages_uv"], spec), spec, calibration=cal))):
            out[mode].append(window_metrics(r["events"], w["sorter_times_s"], w["fs"], w["duration_s"], 1.0,
                                            w["sorter_local_channel"], w["coordinates_um"], RADIUS_UM))
res = {m: {k: pooled(v)[k] for k in ("f1", "recall", "precision", "n_detected")} for m, v in out.items()}
Path(__file__).resolve().parents[2].joinpath("reports/phase5/calibration_transfer_heldout.json").write_text(json.dumps(res, indent=1) + "\n")
print(res)
