"""Regression checks against cached real IBL windows. Skipped with reason 'missing dataset' when the cache is absent."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.phase5.data import load_window, manifest_windows, window_path

pytestmark = pytest.mark.realdata
ROOT = Path(__file__).resolve().parents[1]


def _need(split, k=0):
    try:
        spec = manifest_windows(split)[k]
    except Exception:
        pytest.skip("missing dataset: split manifest not available")
    p = window_path(spec["pid"], spec["t0_s"], spec["duration_s"])
    if not p.exists():
        pytest.skip(f"missing dataset: {p.name} (run scripts/phase5/fetch_all.py)")
    return load_window(p), spec


def test_cached_windows_are_probe_clock_synced():
    w, _ = _need("heldout")
    assert w["sync_version"] >= 3 and w["sync_offset_s"] is not None


def test_legacy_control_reproduces_published_phase4_counts():
    """The original (misaligned) evaluation is reproduced exactly when the original clock mixing is reconstructed."""
    from scripts.phase5.detectors import BASELINE, run_detector
    from scripts.phase5.metrics import legacy_match
    pub = json.loads((ROOT / "reports" / "ibl-detection-localization.json").read_text())["per_window"]
    pid, t0 = "4836a465-c691-4852-a0b1-dcd2b1ce38a1", 100.0
    p = window_path(pid, t0, 1.0)
    if not p.exists():
        pytest.skip(f"missing dataset: {p.name}")
    w = load_window(p)
    ev = run_detector(w["voltages_uv"], BASELINE, w["coordinates_um"])["events"]
    m = legacy_match(ev, w["sorter_times_s"] + w["sync_offset_s"], w["fs"], 1.0)  # reconstruct session-clock times
    o = [x for x in pub if x["t0_s"] == t0][0]["detector_variants"]["car_4sigma"]["vs_sorting"]["1.0"]
    assert (m["true_positives"], m["false_positives"], m["false_negatives"]) == (o["true_positives"], o["false_positives"], o["false_negatives"])
    corrected = legacy_match(ev, w["sorter_times_s"], w["fs"], 1.0)
    assert corrected["true_positives"] > 1.4 * o["true_positives"]


def test_float_fixed_event_agreement_on_real_window():
    from scripts.phase5.compare_float_fixed import spec_from_float
    from scripts.phase5.detectors import run_detector
    from scripts.phase5.fixed_detector import adc_quantize, run_fixed
    from scripts.phase6.mismatch_analyzer import V2
    w, _ = _need("heldout", 1)
    spec = spec_from_float(V2)
    codes = adc_quantize(w["voltages_uv"], spec)
    fl = run_detector(codes * 2.34375, V2, w["coordinates_um"])["events"]
    fx = run_fixed(codes, spec)["events"]
    kf = {(e.sample, e.channel) for e in fl}; kx = {(e.sample, e.channel) for e in fx}
    assert len(kf & kx) / len(kf) > 0.96 and abs(len(kf) - len(kx)) / len(kf) < 0.02
    assert np.array_equal(run_fixed(codes, spec)["car"], run_fixed(codes, spec)["car"])  # deterministic


def test_detection_benchmark_smoke_v2_beats_chance():
    from scripts.phase5.benchmark import evaluate_window
    from scripts.phase6.mismatch_analyzer import V2
    w, spec = _need("selection")
    r = evaluate_window(w, spec, V2, {})
    t = r["tol"]["1.0"]
    assert t["channel_consistent"]["f1"] > 0.3
    assert t["channel_consistent"]["tp"] > 1.5 * t["chance_channel_consistent_tp"]
