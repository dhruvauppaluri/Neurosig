"""Goal C: real-Neuropixels localization stability (proxies only; no ground-truth neuron coordinates).

Usage (numpy/scipy only; reads cached windows):
  .venv/bin/python scripts/phase8/real_stability.py --splits dev      # sessions A-D, calibration+selection windows
  .venv/bin/python scripts/phase8/real_stability.py --splits final    # Phase 6 locked sessions E-H, run ONCE

Events are sorter spikes on the probe clock (Phase 6 clock fix), filtered with the frozen CAR 300-6000 Hz path,
re-aligned to the local trough (+-4 samples) for every method alike. Metrics: within-unit spread (and chance
ratio), channel-dropout displacement and failures, injected-noise displacement, early-vs-late drift, usable coverage.
Sorter peak channels are never used as the optimization target.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from edge.localize import get_localizer  # noqa: E402
from edge.localize.beamform import default_scan_grid_um  # noqa: E402
from edge.localize.template import default_training_xyz_um  # noqa: E402
from edge.phase8.pipeline import Phase8Localizer  # noqa: E402
from edge.phase8.registry import versions  # noqa: E402
from edge.snippets import extract_snippet  # noqa: E402
from edge.types import ElectrodeArray, ElectrodeSite  # noqa: E402
from scripts.phase5.data import load_window, window_path  # noqa: E402
from scripts.real_detector import frozen_real_detector, preprocess  # noqa: E402

N_PRE, N_POST = 30, 45
MAX_EVENTS_PER_WINDOW = 60
METHODS = ("beamform", "template", "robust_scan", "phase8_k1")


def windows_for(splits: str) -> list[dict]:
    p5 = json.loads(Path("reports/phase5/split_manifest.json").read_text())["windows"]
    p6 = json.loads(Path("reports/phase6/split_manifest_phase6.json").read_text())["windows"]
    if splits == "dev":
        return [w for w in p5 if w["split"] in ("calibration", "selection")]
    if splits == "final":
        return list(p6)
    raise ValueError(splits)


def build_array(coords: np.ndarray) -> ElectrodeArray:
    return ElectrodeArray(tuple(ElectrodeSite(i, tuple(float(v) for v in c)) for i, c in enumerate(coords)),
                          "IBL selected contacts (lateral, axial, 0)")


def baseline_engines(array: ElectrodeArray) -> dict:
    y_shift = float(np.min(array.xyz_um()[:, 1]))
    training = default_training_xyz_um(array.n_channels) + np.array([0.0, y_shift, 0.0])
    scan = default_scan_grid_um(array.n_channels) + np.array([0.0, y_shift, 0.0])
    return {
        "beamform": get_localizer("beamform", array=array, grid_xyz_um=scan),
        "template": get_localizer("template", array=array, training_xyz_um=training),
        "robust_scan": get_localizer("robust_scan", array=array, grid_xyz_um=scan, model="line_source"),
    }


def events_from_window(path: Path, seed: int):
    w = load_window(path)
    cfg = frozen_real_detector(w["fs"])
    filt, _ = preprocess(w["voltages_uv"], cfg)
    fs = w["fs"]
    t = w["sorter_times_s"]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(t))
    if len(idx) > MAX_EVENTS_PER_WINDOW:
        idx = np.sort(rng.choice(idx, MAX_EVENTS_PER_WINDOW, replace=False))
    array = build_array(w["coordinates_um"])
    snips, meta = [], []
    for i in idx:
        s0 = int(round(t[i] * fs))
        lo, hi = max(s0 - 4, N_PRE), min(s0 + 5, filt.shape[0] - N_POST - 1)
        if hi <= lo:
            continue
        ch = int(w["sorter_local_channel"][i])
        s = lo + int(np.argmin(filt[lo:hi, ch]))
        snips.append(extract_snippet(filt, s, array, event_id=int(i), sample_rate_hz=fs, n_pre=N_PRE, n_post=N_POST))
        meta.append({"cluster": int(w["sorter_clusters"][i]), "t0_s": w["t0_s"], "pid": w["pid"]})
    return array, snips, meta, w


def mask_snippet(sn, mask):
    v = np.where(mask[None, :], sn.voltages_uv, 0.0)
    return type(sn)(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v,
                    sn.electrode_xyz_um, sn.peak_channel, sn.ground_truth_xyz_um)


def noisy_snippet(sn, k, rng):
    v = np.asarray(sn.voltages_uv, float)
    d = np.diff(v, axis=0)
    sig = np.median(np.abs(d - np.median(d, axis=0)), axis=0) / 0.6745 / np.sqrt(2)
    nv = v + rng.normal(0.0, 1.0, v.shape) * (k * sig)[None, :]
    return type(sn)(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, nv,
                    sn.electrode_xyz_um, sn.peak_channel, sn.ground_truth_xyz_um)


def locate(name, engines, p8, sn, mask=None):
    """Return (xy array or None, extra) for one method; mask applies to phase8 as a weight, baselines see zeros."""
    if name == "phase8_k1":
        an = p8.analyze(sn, mask)
        fit, Wt, sigma = an
        if fit is None:
            return None, None
        s = fit.sources1[0]
        chi2 = fit.sse1 / max(Wt.size - 2 - Wt.shape[1], 1)
        cov = p8.fitter.covariance(np.array([[s.x, s.y]]), Wt, sigma)
        unit_area = float(np.pi * np.sqrt(max(np.linalg.det(0.5 * (cov + cov.T) + 1e-9 * np.eye(2)), 0.0)))
        return np.array([s.x, s.y]), {"chi2": float(chi2), "unit_area": unit_area, "delta": float(fit.delta),
                                      "delta_shared": float(fit.delta_shared)}
    est = engines[name].estimate(sn if mask is None else mask_snippet(sn, mask))
    if not est.available or est.xyz_um is None:
        return None, None
    return np.array(est.xyz_um[:2]), {}


def spread(xy: np.ndarray) -> float:
    med = np.median(xy, axis=0)
    return float(np.median(np.linalg.norm(xy - med, axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", choices=("dev", "final"), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None, help="frozen accept thresholds json (required for final)")
    args = ap.parse_args()
    wins = windows_for(args.splits)
    rng = np.random.default_rng(20268200 if args.splits == "dev" else 20268300)
    frozen = json.loads(Path(args.config).read_text()) if args.config else None
    t_start = time.time()

    rows = []          # one per (event, method)
    drop = []          # dropout displacement rows
    noise = []         # noise displacement rows
    missing = []
    p8_cache: dict = {}
    for wi, w in enumerate(wins):
        path = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not path.exists():
            missing.append(str(path.name))
            continue
        array, snips, meta, win = events_from_window(path, 20268000 + wi)
        if not snips:
            continue
        key = tuple(map(tuple, array.xyz_um().round(1)))
        if key not in p8_cache:
            p8_cache[key] = Phase8Localizer(array, model="line_source", max_x_um=90.0)
        p8 = p8_cache[key]
        engines = baseline_engines(array)
        masks = {lvl: [rng.random(array.n_channels) >= lvl for _ in range(3)] for lvl in (0.25, 0.5)}
        # engines on reduced arrays ("informed" baselines) for dropout
        reduced = {}
        for lvl, ms in masks.items():
            for j, m in enumerate(ms):
                sub = ElectrodeArray(tuple(s for s, keep in zip(array.sites, m) if keep), "reduced")
                reduced[(lvl, j)] = baseline_engines(ElectrodeArray(
                    tuple(ElectrodeSite(i, s.xyz_um) for i, s in enumerate(sub.sites)), "reduced"))
        for sn, mt in zip(snips, meta):
            base_xy = {}
            for name in METHODS:
                xy, extra = locate(name, engines, p8, sn)
                base_xy[name] = xy
                rows.append({"session": w["session"], "pid": mt["pid"][:8], "t0_s": mt["t0_s"], "cluster": mt["cluster"],
                             "method": name, "xy": None if xy is None else xy.tolist(), **(extra or {})})
            for lvl, ms in masks.items():
                j = int(rng.integers(0, 3))
                m = ms[j]
                keep_idx = np.flatnonzero(m)
                sub_sn = type(sn)(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz,
                                  tuple(range(len(keep_idx))), sn.voltages_uv[:, keep_idx], sn.electrode_xyz_um[keep_idx],
                                  None, None)
                for name in METHODS:
                    if base_xy[name] is None:
                        continue
                    if name == "phase8_k1":
                        xy, _ = locate(name, engines, p8, sn, mask=m)
                    else:  # informed baseline: engine built for the reduced contact set
                        e = reduced[(lvl, j)][name].estimate(sub_sn)
                        xy = np.array(e.xyz_um[:2]) if e.available and e.xyz_um is not None else None
                    drop.append({"level": lvl, "method": name, "session": w["session"],
                                 "disp": None if xy is None else float(np.linalg.norm(xy - base_xy[name]))})
            for k in (0.5, 1.0):
                nsn = noisy_snippet(sn, k, rng)
                for name in METHODS:
                    if base_xy[name] is None:
                        continue
                    xy, _ = locate(name, engines, p8, nsn)
                    noise.append({"k": k, "method": name, "session": w["session"],
                                  "disp": None if xy is None else float(np.linalg.norm(xy - base_xy[name]))})
        print(f"window {wi + 1}/{len(wins)} {w['session']} {path.name}: {len(snips)} events, {time.time() - t_start:.0f}s",
              flush=True)

    # ---- accept rule for phase8 (thresholds from dev; frozen file for final) ----
    p8_rows = [r for r in rows if r["method"] == "phase8_k1"]
    if frozen is None:
        chis = np.array([r["chi2"] for r in p8_rows]); ar = np.array([r["unit_area"] for r in p8_rows])
        frozen = {"max_chi2": float(np.percentile(chis, 95)), "max_unit_area_um2": float(np.percentile(ar, 95)),
                  "derived_from": "dev windows (P95 of K=1 chi2 and unit area)"}
    for r in p8_rows:
        r["accepted"] = bool(r["chi2"] <= frozen["max_chi2"] and r["unit_area"] <= frozen["max_unit_area_um2"])
    for r in rows:
        r.setdefault("accepted", r["xy"] is not None)

    report = {"versions": versions(), "splits": args.splits, "n_windows": len(wins), "missing_windows": missing,
              "frozen_accept": frozen, "methods": {}, "limitations": [
                  "Sorter-derived positions and repeated-unit consistency are proxies, not neuron coordinates.",
                  "Stability metrics reward consistency; a constant output would also be consistent, so the chance ratio is reported.",
                  "phase8_k1 uses single-source mode on real data; two-source declarations are not validated here.",
              ]}
    for name in METHODS:
        rr = [r for r in rows if r["method"] == name]
        usable = [r for r in rr if r["xy"] is not None and r["accepted"]]
        by_unit = defaultdict(list)
        for r in usable:
            by_unit[(r["pid"], r["cluster"])].append(r)
        units = {k: np.array([r["xy"] for r in v]) for k, v in by_unit.items() if len(v) >= 5}
        within = [spread(v) for v in units.values()]
        wy = [float(np.median(np.abs(v[:, 1] - np.median(v[:, 1])))) for v in units.values()]
        allxy = np.vstack(list(units.values())) if units else np.zeros((0, 2))
        total_y = float(np.median(np.abs(allxy[:, 1] - np.median(allxy[:, 1])))) if len(allxy) else None
        # early vs late drift within units
        drift = []
        for k, v in by_unit.items():
            ts = np.array([r["t0_s"] for r in v])
            if len(v) < 6 or np.ptp(ts) <= 0:
                continue
            xy = np.array([r["xy"] for r in v]); cut = np.median(ts)
            e, l = xy[ts <= cut], xy[ts > cut]
            if len(e) >= 2 and len(l) >= 2:
                drift.append(float(np.linalg.norm(np.median(e, axis=0) - np.median(l, axis=0))))
        # per-session within spread for cross-session consistency
        per_sess = {}
        for s in sorted({r["session"] for r in usable}):
            sp = [spread(np.array([r["xy"] for r in v])) for k, v in by_unit.items()
                  if len(v) >= 5 and v[0]["session"] == s]
            per_sess[s] = {"n_units": len(sp), "median_within_unit_spread_um": float(np.median(sp)) if sp else None}
        d = {lvl: [x for x in drop if x["method"] == name and x["level"] == lvl] for lvl in (0.25, 0.5)}
        nz = {k: [x for x in noise if x["method"] == name and x["k"] == k] for k in (0.5, 1.0)}

        def disp_stats(items):
            v = np.array([x["disp"] for x in items if x["disp"] is not None])
            return {"n": len(items), "failure_rate": 1 - len(v) / max(len(items), 1),
                    "median_disp_um": float(np.median(v)) if len(v) else None,
                    "p90_disp_um": float(np.percentile(v, 90)) if len(v) else None}
        report["methods"][name] = {
            "events": len(rr), "usable_coverage": len(usable) / max(len(rr), 1),
            "n_units_ge5": len(units),
            "within_unit_spread_um_median": float(np.median(within)) if within else None,
            "within_unit_spread_um_mean": float(np.mean(within)) if within else None,
            "within_unit_axial_mad_um_median": float(np.median(wy)) if wy else None,
            "total_axial_mad_um": total_y,
            "chance_ratio_axial": (float(np.median(wy)) / total_y) if wy and total_y else None,
            "early_late_drift_um_median": float(np.median(drift)) if drift else None, "n_drift_units": len(drift),
            "dropout": {str(k): disp_stats(v) for k, v in d.items()},
            "noise": {str(k): disp_stats(v) for k, v in nz.items()},
            "per_session": per_sess,
        }
    out = Path(args.out or f"reports/phase8/real_stability_{args.splits}.json")
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    for name, m in report["methods"].items():
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                     if k in ("events", "usable_coverage", "n_units_ge5", "within_unit_spread_um_median",
                              "within_unit_axial_mad_um_median", "chance_ratio_axial", "early_late_drift_um_median")})
        print("   dropout", m["dropout"], "\n   noise", m["noise"])
    if args.splits == "dev":
        Path("reports/phase8/frozen_real_accept.json").write_text(json.dumps(frozen, indent=2) + "\n")


if __name__ == "__main__":
    main()
