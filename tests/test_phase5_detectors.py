"""Phase 5 float detector: baseline equivalence and metric sanity checks."""

import numpy as np

from scripts.phase5.detectors import BASELINE, P5Config, run_detector
from scripts.phase5.metrics import match_optimal, pooled, window_metrics
from scripts.real_detector import detect_real, frozen_real_detector


def synthetic(seed=0, n=30000, ch=16):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 8.0, (n, ch)) + rng.normal(0, 5.0, (n, 1))  # noise + common mode
    truth = []
    for s in range(1500, n - 200, 900):
        c = int(rng.integers(0, ch))
        t = np.arange(-20, 40)
        x[s + t, c] += -90.0 * np.exp(-0.5 * (t / 4.0) ** 2) + 25.0 * np.exp(-0.5 * ((t - 14) / 7.0) ** 2)
        truth.append(s / 30000.0)
    return x, np.array(truth)


def test_baseline_config_reproduces_frozen_detector_exactly():
    x, _ = synthetic()
    frozen = detect_real(x, frozen_real_detector(30000.0))["events"]
    mine = run_detector(x, BASELINE)["events"]
    assert [(e.sample, e.channel) for e in mine] == [(e.sample, e.channel) for e in frozen]
    assert len(mine) > 5


def test_optimal_match_is_one_to_one_and_within_tolerance():
    pairs = match_optimal(np.array([100, 103, 500]), np.array([101, 104, 900]), 3)
    assert sorted((a, b) for a, b, _ in pairs) == [(0, 0), (1, 1)]
    assert all(abs(dt) <= 3 for _, _, dt in pairs)


def test_metrics_counts_add_up():
    x, truth = synthetic()
    ev = run_detector(x, BASELINE)["events"]
    m = window_metrics(ev, truth, 30000.0, 1.0, 1.0)
    assert m["tp"] + m["fp"] == m["n_detected"]
    assert m["tp"] + m["fn"] == m["n_reference"]
    assert pooled([m])["tp"] == m["tp"]


def test_each_variant_runs_and_is_deterministic():
    x, _ = synthetic()
    coords = np.stack([np.zeros(16), np.arange(16) * 20.0, np.zeros(16)], axis=1)
    for cfg in (BASELINE.replace(car="mean"), BASELINE.replace(car="trimmed"), BASELINE.replace(car="none"),
                BASELINE.replace(noise="mad"), BASELINE.replace(noise="rms_clip"), BASELINE.replace(noise="iqr"),
                BASELINE.replace(noise="ema"), BASELINE.replace(merge="nms", merge_radius_um=60.0),
                BASELINE.replace(min_width_samples=2), BASELINE.replace(delay_comp_samples=-3)):
        a = run_detector(x, cfg, coords)["events"]
        b = run_detector(x, cfg, coords)["events"]
        assert a == b
