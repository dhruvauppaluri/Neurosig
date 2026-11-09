"""Phase 14.4: detector-event caches for online unit association (no sorter output is used to build features).

For every window of a session, the frozen detector-v2.0 (Phase 5 float reference, 200-4500 Hz, median CAR, 4 sigma)
is run on the raw AP window. Per detector event we keep:
  wave     [45, C] frozen-preprocessing (300-6000 Hz, median CAR) waveform, samples -15..+29 around the detector
           trough, divided by that window's per-contact MAD noise (float16)
  amp3     [C] E3-style trough amplitudes on raw - median CAR (mean of 3 samples at the E3-aligned trough, baseline
           removed), the single-source localization input of P12
  sigma_rc [C] raw-CAR noise of the window (for FastLine20 weights)
  t, ch, snr, win
Evaluation-only fields (never read by the grouper): ``sorter_unit`` of the optimally matched sorter event
(|dt| <= 0.5 ms, channel-consistent; -1 = unmatched) and the sorter unit's quality label / depth.
Cached as data/phase14/det_<session>.npz.
"""

from __future__ import annotations

import numpy as np

from edge.phase11.referencing import align_sample_raw, event_quality, load_events, match_ab, raw_car
from scripts.phase5.data import load_window, window_path

WAVE = (-15, 30)


def build(session: str, wins: list[dict], out_path, log=print) -> dict:
    cols = {k: [] for k in ("wave", "amp3", "t", "ch", "snr", "win", "sorter_unit", "sample")}
    sig_rc, E = [], None
    for wi, w in enumerate(wins):
        p = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not p.exists():
            log(f"missing window {p}"); continue
        we = load_events(w, load_window(p))
        E = we.array.xyz_um()[:, :2]
        rc, sig = raw_car(we)
        d2s = {d: r for d, r, _ in match_ab(we, we.det_sample)}
        for j in np.argsort(we.det_sample, kind="stable"):
            s, ch = int(we.det_sample[j]), int(we.det_channel[j])
            if s - 45 < 0 or s + 45 > we.raw.shape[0]:
                continue
            s3 = align_sample_raw(we, s, ch, 3)
            base = np.median(rc[s3 - 40:s3 - 12], axis=0)
            cols["amp3"].append((-(rc[s3 - 1:s3 + 2].mean(axis=0) - base)).astype(np.float32))
            cols["wave"].append((we.align[s + WAVE[0]:s + WAVE[1]] / we.sigma[None]).astype(np.float16))
            cols["t"].append(float(w["t0_s"]) + s / we.fs); cols["ch"].append(ch); cols["win"].append(wi)
            cols["snr"].append(event_quality(we, s, ch)["snr"]); cols["sample"].append(s)
            cols["sorter_unit"].append(int(we.sorter_cluster[d2s[int(j)]]) if int(j) in d2s else -1)
        sig_rc.append(sig.astype(np.float32))
        log(f"{session} window {wi + 1}/{len(wins)} det events {len(cols['t'])}")
    order = np.argsort(np.array(cols["t"]), kind="stable")
    out = {k: np.asarray(v)[order] for k, v in cols.items()}
    np.savez_compressed(out_path, E=E, sigma_rc=np.stack(sig_rc), **out)
    return {"n_events": int(len(order)), "matched_frac": float(np.mean(out["sorter_unit"] >= 0)) if len(order) else 0.0}


def load(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
