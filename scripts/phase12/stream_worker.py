"""Per-window worker for the Phase 12.6 stream benchmark (importable by multiprocessing)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")


def process_window(w):
    from edge.phase10.overlap_stress import frozen_calibration
    from edge.phase11.referencing import load_events, raw_car
    from edge.phase12 import count_model as CM, fast as FA
    from edge.snippets import extract_snippet
    from scripts.phase5.data import load_window, window_path
    CAL = frozen_calibration()
    LR = CM.LogisticCount.from_json(json.loads(Path("reports/phase12/count_model_p12.json").read_text())["logistic"])
    we = load_events(w, load_window(window_path(w["pid"], w["t0_s"], w["duration_s"])))
    E = we.array.xyz_um(); loc = FA.fast_localizer(we.array); fl = FA.FastLine20(E)
    t0 = time.perf_counter()
    rc, sig = raw_car(we)
    busy = time.perf_counter() - t0
    lat, single, two, arr = [], [], [], []
    for j in np.argsort(we.det_sample, kind="stable"):
        s, ch = int(we.det_sample[j]), int(we.det_channel[j])
        if s - 60 < 0 or s + 70 > rc.shape[0]:
            continue
        t = time.perf_counter()
        s3 = FA.e3_align(rc, s, ch, E, sig)
        th, ar, f = FA.cheap_features(we.raw[s3 - 40:s3 + 60], 40, fl, sig)
        single.append((time.perf_counter() - t) * 1e3)
        a = CM.analyze_fast(loc, extract_snippet(we.raw, s3, we.array, event_id=int(j), sample_rate_hz=we.fs))
        two.append(bool(a is not None and CM.decide_lr(a, CAL, LR).n >= 2))
        dt = time.perf_counter() - t
        lat.append(dt * 1e3); busy += dt; arr.append(s / we.fs)
    return {"lat": np.array(lat), "single_ms": np.array(single), "two": np.array(two), "arr_s": np.array(arr),
            "dur_s": we.raw.shape[0] / we.fs, "busy_s": busy}
