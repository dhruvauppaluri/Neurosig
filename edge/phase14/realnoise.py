"""Phase 14: per-session event-free background segments from raw IBL windows (sorter-free by >= 3 ms), in both the
window's native channel order (for the session NoiseModel applied to real snippets) and axial order (for label-free
threshold self-calibration with development-morphology templates, bench14.generate(calibration=True,
noise_override=...)), plus the frozen-preprocessing MAD per contact in axial order (SNR scale of the calibration)."""
from __future__ import annotations

import numpy as np

SEG = 120


def session_noise(wins: list[dict], n: int = 3000, seed: int = 0):
    from scripts.phase5.data import load_window, window_path
    from scripts.real_detector import frozen_real_detector, preprocess
    rng = np.random.default_rng(seed); segs = []; mads = []; order = None
    per = n // max(len(wins), 1) + 1
    for w in wins:
        q = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not q.exists():
            continue
        d = load_window(q); v = np.asarray(d["voltages_uv"], float); fs = d["fs"]
        order = np.argsort(np.asarray(d["coordinates_um"])[:, 1], kind="stable")
        ok = []
        for margin in (90, 45, 30):          # dense recordings: relax the sorter-free margin (recorded below)
            busy = np.zeros(v.shape[0], bool)
            for s in np.rint(np.asarray(d["sorter_times_s"]) * fs).astype(int):
                busy[max(s - margin, 0):s + margin] = True
            ok = [s for s in range(0, v.shape[0] - SEG, 7) if not busy[s:s + SEG].any()]
            if len(ok) >= 20:
                break
        if not ok:
            continue
        for s in rng.choice(ok, min(per, len(ok)), replace=False):
            segs.append(v[s:s + SEG].astype(np.float32))
        f, _ = preprocess(v, frozen_real_detector(fs))
        mads.append(np.median(np.abs(f - np.median(f, axis=0)), axis=0) / 0.6745)
    native = np.stack(segs)[:n]
    return native, native[:, :, order], np.median(np.stack(mads), axis=0)[order]
