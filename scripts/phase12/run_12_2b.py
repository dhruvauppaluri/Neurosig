"""Phase 12.2b: sorter-free online grouping of detector-v2.0 events, then unit-level pooling of the groups.
Usage: run_12_2b.py dev|select. Purity uses sorter labels of matched events only (evaluation, not grouping)."""
import json, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase11.configs import p11_factory
from edge.phase11.referencing import align_sample_raw, event_quality, load_events, match_ab, raw_car
from edge.phase12 import pooling as PL
from edge.phase12.online_groups import MIN_GROUP, OnlineGrouper
from scripts.phase5.data import load_window, window_path
from edge.phase12.splits import split_windows

split = sys.argv[1]
RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
by = defaultdict(list)
for w in split_windows(split):
    by[w["session"]].append(w)
t0 = time.time()
out = {"versions": versions(), "split": split, "rho": RHO, "sessions": {}}
rows_all = []
for sess, ws in sorted(by.items()):
    grp = None; cols = defaultdict(list); array = None
    for wi, w in enumerate(ws):
        p = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not p.exists():
            continue
        we = load_events(w, load_window(p)); array = we.array
        if grp is None:
            grp = OnlineGrouper(we.array.xyz_um(), rho=RHO)
        rc, sig = raw_car(we)
        d2s = {d: r for d, r, _ in match_ab(we, we.det_sample)}
        for j in np.argsort(we.det_sample, kind="stable"):
            s, ch = int(we.det_sample[j]), int(we.det_channel[j])
            if s + PL.SEG[0] - 2 < 0 or s + PL.SEG[1] + 2 > we.raw.shape[0]:
                continue
            g = grp.add(rc, s, ch, (wi, int(j)))
            if g is None:
                continue
            s3 = align_sample_raw(we, s, ch, 3)
            cols["seg"].append(we.raw[s + PL.SEG[0]:s + PL.SEG[1]].astype(np.float32)); cols["shift"].append(s3 - s)
            cols["t"].append(float(w["t0_s"]) + s / we.fs); cols["gid"].append(g); cols["ch"].append(ch)
            cols["snr"].append(event_quality(we, s3, ch)["snr"]); cols["win"].append(wi)
            cols["sorter_unit"].append(int(we.sorter_cluster[d2s[int(j)]]) if int(j) in d2s else -1)
        cols["sigw"].append(sig.astype(np.float32))
    gid = np.array(cols["gid"]); su = np.array(cols["sorter_unit"])
    sizes = Counter(gid.tolist())
    assigned = np.array([sizes[g] >= MIN_GROUP for g in gid])
    pur_num = pur_den = 0; groups_per_unit = defaultdict(set)
    for g in {g for g in gid[assigned]}:
        m = su[(gid == g) & (su >= 0)]
        if len(m):
            c = Counter(m.tolist()).most_common(1)[0][1]; pur_num += c; pur_den += len(m)
            for u in set(m.tolist()):
                if np.sum(m == u) >= 3:
                    groups_per_unit[u].add(g)
    order = np.argsort(np.array(cols["t"]), kind="stable")
    se = PL.SessionEvents(sess, array, array.xyz_um(), 30000.0, np.stack(cols["seg"])[order], np.stack(cols["sigw"]),
                          np.array(cols["shift"])[order], np.array(cols["t"])[order], gid[order], np.array(cols["ch"])[order],
                          np.array(cols["snr"])[order], (su >= 0)[order], np.array(cols["win"])[order])
    ul = PL.UnitLocalizer(se, p11_factory(array)["weighted_scan"])
    units = {int(g): np.flatnonzero(se.unit == g) for g in sorted(set(gid[assigned].tolist()))}
    rows = PL.unit_stability(ul, units, seed=1, ns=(1, 5, 10, 25, 50))
    rows_all += rows
    out["sessions"][sess] = {"detector_events": int(len(gid)), "assigned_fraction": float(assigned.mean()),
                             "n_groups_assigned": len(units), "n_groups_tentative": int(sum(1 for v in sizes.values() if v < MIN_GROUP)),
                             "purity_matched_events": pur_num / max(pur_den, 1), "matched_events_in_assigned": int(pur_den),
                             "sorter_units_split_median": float(np.median([len(v) for v in groups_per_unit.values()])) if groups_per_unit else None,
                             "assigned_fraction_snr_ge8": float(assigned[np.array(cols["snr"]) >= 8].mean()) if np.any(np.array(cols["snr"]) >= 8) else None}
    print(sess, out["sessions"][sess], f"{time.time() - t0:.0f}s", flush=True)
keys = ["ws_single", "ws_mean", "ws_robust", "l20_single", "l20_mean", "l20_robust", "l20_joint"]
out["summary"] = PL.summarize_stability(rows_all, keys, fixed_min_events=50)
out["runtime_s"] = time.time() - t0
Path(f"reports/phase12/exp12_2b_{split}_rho{RHO:g}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
pickle.dump(rows_all, open(f"data/phase12/rows12_2b_{split}_rho{RHO:g}.pkl", "wb"))
for N, o in out["summary"].items():
    print(N, o["n_units"], {k: round(v["median"], 1) for k, v in o.items() if isinstance(v, dict)})
