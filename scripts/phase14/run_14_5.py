"""Phase 14.5: identity maintenance and position tracking under simulated probe drift (ground truth known).

Usage: .venv/bin/python scripts/phase14/run_14_5.py <pool> [n_seeds]
  pool: p14dev (development morphologies) | p14select | p14final (unseen morphologies; run_final14.py only)
For each drift profile and seed: the frozen online association (OU-P14 parameters from frozen_p14.json, or the dev
defaults before freezing) runs on the stream; identity is scored against the TRUE neuron labels; positions are
tracked per true unit and per online unit by static / window / EMA / Kalman / Kalman + change-point trackers.
Tracking error = |track - truth - bias| on the axial coordinate, where bias = median(track - truth) over the first
10% of each unit's events (localization bias of the single-event fit is constant per unit and is reported
separately); scored on the remaining 90%. Writes reports/phase14/exp14_5_<pool>.json.
"""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase12.fast import FastLine20
from edge.phase14 import drift_sim as DS
from edge.phase14.online_units import OnlineUnits, Params, evaluate

PROFILES = [("none", 0.0), ("linear", 30.0), ("step", 25.0), ("sine", 20.0)]
TRACKERS = [("static", {}), ("window", {"W": 30}), ("ema", {"alpha": 0.05}), ("kalman", {"q": 0.5, "r": 15.0}),
            ("kalman", {"q": 2.0, "r": 15.0}), ("kalman_cp", {"q": 0.5, "r": 15.0, "h": 8.0})]


def params():
    p = Path("reports/phase14/frozen_p14.json")
    if p.exists():
        return Params(**json.loads(p.read_text())["online_units"])
    return Params(min_gain=80, merge_every=10 ** 9, snr_assign=6.0)


def tracking_error(t, P, truth, method, kw):
    tr = DS.track_positions(t, P, method, **kw)
    n0 = max(int(0.1 * len(t)), 5)
    bias = np.median(tr[:n0, 1] - truth[:n0, 1])
    e = np.abs(tr[n0:, 1] - truth[n0:, 1] - bias)
    return e, float(bias)


def main():
    pool = sys.argv[1]; n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    prm = params()
    for a in sys.argv[3:]:
        k, v = a.split("="); setattr(prm, k, type(getattr(prm, k))(float(v)))
    tag = "_".join(sys.argv[3:])
    out = {"versions": versions(), "pool": pool, "params": prm.__dict__, "runs": []}
    t0 = time.time()
    for drift, amp in PROFILES:
        for seed in range(n_seeds):
            st = DS.make_stream(pool, drift=drift, amp_um=amp, seed=1000 * (1 + PROFILES.index((drift, amp))) + seed)
            ou = OnlineUnits(st["E"], prm)
            for i in range(len(st["t"])):
                ou.add(st["wave"][i].astype(float), st["amp3"][i], int(st["ch"][i]), float(st["t"][i]), float(st["snr"][i]), i)
            lab = ou.final_labels()
            ev = evaluate(lab, st["unit"], st["snr"])
            q = len(st["t"]) // 4
            ev["coverage_first_quarter"] = float(np.mean(lab[:q] >= 0)); ev["coverage_last_quarter"] = float(np.mean(lab[-q:] >= 0))
            fl = FastLine20(np.c_[st["E"], np.zeros(len(st["E"]))])
            fits = [fl.fit(a, st["sigma"], iters=5)[0] for a in st["amp3"]]
            ok = np.array([f is not None for f in fits])
            P = np.array([f if f is not None else [np.nan, np.nan] for f in fits])
            run = {"drift": drift, "amp_um": amp, "seed": seed, "n_events": int(len(st["t"])), "n_units": int(len(set(st["unit"].tolist()))),
                   "identity": {k: ev[k] for k in ("precision", "coverage_all", "merge_rate", "fragmentation_mean", "duplicate_units_frac", "ari", "n_units")},
                   "coverage_first_quarter": ev["coverage_first_quarter"], "coverage_last_quarter": ev["coverage_last_quarter"],
                   "fit_unavailable": float(1 - ok.mean()), "tracking": {}}
            for name, kw in TRACKERS:
                key = name + ("_" + "_".join(f"{k}{v}" for k, v in kw.items()) if kw else "")
                errs, errs_online, bias = [], [], []
                for u in np.unique(st["unit"]):
                    m = (st["unit"] == u) & ok
                    if m.sum() < 30:
                        continue
                    e, b = tracking_error(st["t"][m], P[m], st["true_pos"][m], name, kw)
                    errs += e.tolist(); bias.append(b)
                    # online identity: events of the online unit holding most of this true unit's events
                    lu = lab[m]
                    if (lu >= 0).sum() >= 30:
                        v, c = np.unique(lu[lu >= 0], return_counts=True); g = v[np.argmax(c)]
                        mm = (lab == g) & ok
                        e2, _ = tracking_error(st["t"][mm], P[mm], st["true_pos"][mm], name, kw)
                        errs_online += e2.tolist()
                run["tracking"][key] = {"median_um": float(np.median(errs)) if errs else np.nan, "p90_um": float(np.percentile(errs, 90)) if errs else np.nan,
                                        "online_median_um": float(np.median(errs_online)) if errs_online else np.nan,
                                        "bias_median_um": float(np.median(np.abs(bias))) if bias else np.nan}
            out["runs"].append(run)
            print(drift, seed, {k: round(v, 3) if isinstance(v, float) else v for k, v in run["identity"].items()},
                  {k: round(v["median_um"], 1) for k, v in run["tracking"].items()}, f"{time.time() - t0:.0f}s", flush=True)
    summ = {}
    for drift, _ in PROFILES:
        R = [r for r in out["runs"] if r["drift"] == drift]
        summ[drift] = {"precision": float(np.median([r["identity"]["precision"] for r in R])),
                       "fragmentation": float(np.median([r["identity"]["fragmentation_mean"] for r in R])),
                       "coverage_last_quarter": float(np.median([r["coverage_last_quarter"] for r in R])),
                       "tracking_median_um": {k: float(np.median([r["tracking"][k]["median_um"] for r in R])) for k in R[0]["tracking"]},
                       "tracking_online_median_um": {k: float(np.nanmedian([r["tracking"][k]["online_median_um"] for r in R])) for k in R[0]["tracking"]}}
    out["summary"] = summ
    Path(f"reports/phase14/exp14_5_{pool}{'_' + tag if tag else ''}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    for k, v in summ.items():
        print(k, round(v["precision"], 3), round(v["fragmentation"], 2), {kk: round(vv, 1) for kk, vv in v["tracking_median_um"].items()})


if __name__ == "__main__":
    main()
