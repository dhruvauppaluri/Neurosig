"""Phase 11G: staged pipeline gate calibration (p11dev) and accuracy/latency on p11select vs the always-full K<=2
pipeline; throughput at the measured real event rate; P10 mask-cache memory bound. Single thread, this machine."""
import argparse, json, pickle, platform, sys, time, tracemalloc
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase10.configs import p10_factory
from edge.phase10.overlap_stress import localizer, match, summarize
from edge.phase9.improved import condition, dead_channels
from edge.phase11.overlap import CountRule, localize as count_localize
from edge.phase11.staged import AmbiguityScore, StagedLocalizer

RULE = CountRule(axial_only_boundary=True)
_ap = argparse.ArgumentParser(); _ap.add_argument("--mode", default="greedy2"); MODE = _ap.parse_args().mode
array, dev = pickle.load(open("data/phase11/stress_p11dev.pkl", "rb"))
_, sel = pickle.load(open("data/phase11/stress_p11select.pkl", "rb"))
loc = localizer(array); sc = AmbiguityScore(array, mode=MODE)

def full(e):
    c = condition(e.snippet, baseline=True); r, _ = count_localize(loc, c, RULE); return r

# gate calibration on p11dev: pass >= 98% of events the full pipeline declares two-source
scores, two = [], []
for fam, evs in dev.items():
    for e in evs:
        r = full(e); s, _ = sc(e.snippet)
        scores.append(s); two.append(r.n_sources >= 2)
scores, two = np.array(scores), np.array(two)
tau = float(np.percentile(scores[two], 2))
print("tau_gate", round(tau, 2), "dev invoked fraction", round(float(np.mean(scores > tau)), 3), flush=True)

st = StagedLocalizer(array, tau, mode=MODE)
rows = {"staged": [], "full": []}
for fam, evs in sel.items():
    for e in evs:
        truth = np.array([s[:2] for s in e.sources])
        t = time.perf_counter(); rf = full(e); tf = (time.perf_counter() - t) * 1e3
        o = st.localize(e.snippet)
        if o["refined"] is not None:
            r = o["refined"]; est = np.array([s.xyz_um[:2] for s in r.sources]) if r.sources else np.zeros((0, 2)); n, acc = r.n_sources, r.accepted
        else:
            est = np.array([o["fast"].xyz_um[:2]]) if o["fast"].available else np.zeros((0, 2)); n, acc = len(est), o["fast"].available
        estf = np.array([s.xyz_um[:2] for s in rf.sources]) if rf.sources else np.zeros((0, 2))
        for key, ee, nn, aa, ms in (("staged", est, n, acc, o["ms"]), ("full", estf, rf.n_sources, rf.accepted, tf)):
            m = match(truth, ee)
            rows[key].append({"kind": e.kind, "res": e.resolvable, "dt": e.dt, "n_est": nn, "accepted": aa, "flags": [], "ms": ms,
                              "invoked": o["invoked"], **m})
out = {"machine": platform.platform(), "mode": MODE, "tau_gate": tau, "gate_rule": "2nd percentile of the score on p11dev events the full pipeline declares two-source"}
for key, R in rows.items():
    pairs = [r for r in R if r["kind"] == "pair"]; res = [r for r in pairs if r["res"]]; sing = [r for r in R if r["kind"] == "single"]
    ms = np.array([r["ms"] for r in R])
    out[key] = {"resolvable_recall": summarize(res, 2)["all_sources_recovered"], "false_two_singles": float(np.mean([r["n_est"] >= 2 for r in sing])),
                "simultaneous_recall": summarize([r for r in res if r["dt"] == 0], 2)["all_sources_recovered"],
                "matched_err_median": summarize(res, 2)["matched_err_median_um"],
                "latency_ms": {"median": float(np.median(ms)), "mean": float(ms.mean()), "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99))},
                "single_event_latency_ms_median": float(np.median([r["ms"] for r in sing])),
                "invoked_fraction": float(np.mean([r["invoked"] for r in R])) if key == "staged" else 1.0,
                "invoked_fraction_singles": float(np.mean([r["invoked"] for r in sing])) if key == "staged" else 1.0}
    print(key, json.dumps(out[key]), flush=True)
# real event rate (dev detector events per second per 32-contact probe section) and sustainable throughput
ev_rate = 12625 / (24 * 2.0)
out["real_event_rate_per_s"] = ev_rate
for key in ("staged", "full"):
    out[key]["sustainable_events_per_s_single_core"] = 1000.0 / out[key]["latency_ms"]["mean"]
    out[key]["margin_vs_real_rate"] = out[key]["sustainable_events_per_s_single_core"] / ev_rate
# P10 mask-cache memory bound: 40 distinct dead-contact masks with cache sizes 16 (default) and 4
from edge.phase10 import configs as C10
from edge.types import Snippet
sn = sel["single"][0].snippet
for size in (16, 4):
    tracemalloc.start()
    eng = {k: C10.DeadAwareLocalizer(array, k, e, size) for k, e in __import__("edge.phase9.configs", fromlist=["p9_factory"]).p9_factory(array).items()}
    base = tracemalloc.get_traced_memory()[0]
    rng = np.random.default_rng(0)
    for i in range(40):
        v = np.asarray(sn.voltages_uv, float).copy(); v[:, rng.choice(32, 6, replace=False)] = 0.0
        z = Snippet(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v, sn.electrode_xyz_um, sn.peak_channel, None)
        for e in eng.values():
            e.estimate(z)
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    out[f"p10_cache{size}_resident_mib_after_40_masks"] = (cur - base) / 2 ** 20
    out[f"p10_cache{size}_peak_mib"] = (peak - base) / 2 ** 20
    print("cache", size, round((cur - base) / 2 ** 20, 1), "MiB", flush=True)
Path(f"reports/phase11/exp11g_runtime_{MODE}.json").write_text(json.dumps(out, indent=1) + "\n")
