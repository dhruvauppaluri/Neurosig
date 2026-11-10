"""Phase 15 Track C identity experiments on detector-event caches (sorter labels for evaluation only).

Usage: .venv/bin/python scripts/phase15/run_identity.py <split> '<json: params / lookback_s / llr_min / snr_assign ...>' <tag>
  split: dev (A-C) | select (D-S) | long_dev / long_select (30 s contiguous windows, data/phase15/det_long_<s>.npz)
         | final (Phase 15 sessions; requires frozen_p15.json)
Writes reports/phase15/id_<split>_<tag>.json: per-session metrics (edge/phase14/online_units.evaluate), abstention
breakdown, and the breakdown restricted to matched high-SNR events.
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
from edge.phase14 import detcache as DC
from edge.phase14.online_units import Params, evaluate
from edge.phase15.identity import OnlineUnits15
from scripts.phase14.run_14_4 import quality_map, unit_localization
from edge.phase12.fast import FastLine20

SPLITS = {"dev": list("ABC"), "select": list("DEFGHIJKLMNOPQRS"), "long_dev": list("ABC"), "long_select": list("DLO"), "long": list("ABDLO")}


def path_for(split, s):
    if split.startswith("long"):
        return Path(f"data/phase15/det_long_{s}.npz")
    if split == "final":
        return Path(f"data/phase15/det_{s}.npz")
    if split == "final_long":
        return Path(f"data/phase15/det_long_{s}.npz")
    return Path(f"data/phase14/det_{s}.npz")


def run_session(split, s, cfg):
    p = path_for(split, s)
    if not p.exists():
        return None
    c = DC.load(p)
    kw = {k: v for k, v in cfg.items() if k in Params.__dataclass_fields__}
    base = dict(min_gain=80.0, merge_every=10 ** 9, snr_assign=6.0); base.update(kw)
    ou = OnlineUnits15(c["E"], Params(**base), lookback_s=cfg.get("lookback_s", 0.0), llr_min=cfg.get("llr_min", np.inf),
                       low_snr_floor=cfg.get("low_snr_floor", 4.0), tau_rel=cfg.get("tau_rel", 0.0),
                       refractory_ms=cfg.get("refractory_ms", 0.0), peak_um=cfg.get("peak_um", np.inf), margin_rel=cfg.get("margin_rel", 0.0),
                       split_every=cfg.get("split_every", 0), split_dprime=cfg.get("split_dprime", 4.0))
    t = time.perf_counter()
    for i in range(len(c["t"])):
        ou.add(c["wave"][i].astype(float), c["amp3"][i], int(c["ch"][i]), float(c["t"][i]), float(c["snr"][i]), i)
    ms = (time.perf_counter() - t) * 1e3 / max(len(c["t"]), 1)
    lab = ou.final_labels()
    qm = quality_map(s)
    if qm is None and Path("data/phase15/cluster_info15.pkl").exists():
        import pickle
        d15 = pickle.load(open("data/phase15/cluster_info15.pkl", "rb"))
        qm = dict(zip(d15[s]["cluster_id"], d15[s]["label"])) if s in d15 else None
    qual = np.array([qm.get(int(u), -1) if (qm and u >= 0) else -1 for u in c["sorter_unit"]]) if qm else None
    ev = evaluate(lab, c["sorter_unit"], c["snr"], qual)
    ev["ms_per_event"] = ms
    br, cats = ou.reasons_final()
    ev["abstention_all"] = br
    hq = (c["sorter_unit"] >= 0) & (c["snr"] >= 8) & ((qual >= 1) if qual is not None else True)
    v, n = np.unique(cats[hq], return_counts=True)
    ev["abstention_hq"] = {k: float(x / max(hq.sum(), 1)) for k, x in zip(v, n)}
    m = c["sorter_unit"] >= 0
    v, n = np.unique(cats[m], return_counts=True)
    ev["abstention_matched"] = {k: float(x / max(m.sum(), 1)) for k, x in zip(v, n)}
    fl = FastLine20(np.c_[c["E"], np.zeros(len(c["E"]))]); sigma = np.median(c["sigma_rc"], axis=0).astype(float)
    loc = unit_localization(c, lab, fl, sigma)
    if loc:
        es = np.nanmedian([r["event_spread"] for r in loc]); us = np.nanmedian([r["unit_spread"] for r in loc])
        ev["spread_reduction"] = float(1 - us / es)
    ev["duration_s"] = float(c["t"].max() - c["t"].min())
    return ev


def main():
    split, cfg, tag = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
    if split in ("final", "final_long"):
        if not Path("reports/phase15/frozen_p15.json").exists():
            raise SystemExit("frozen required")
        sessions = list(json.loads(Path("reports/phase15/split_manifest_phase15.json").read_text())["sessions"])
    else:
        sessions = SPLITS[split]
    out = {"split": split, "cfg": cfg, "sessions": {}}
    for s in sessions:
        e = run_session(split, s, cfg)
        out["sessions"][s] = e if e is not None else {"status": "missing_dataset"}
    ok = [v for v in out["sessions"].values() if "precision" in v]
    keys = ("precision", "precision_hq", "coverage_hq", "coverage_all", "coverage_matched", "merge_rate", "fragmentation_mean", "ari", "spread_reduction", "ms_per_event")
    out["summary"] = {k: float(np.nanmedian([v.get(k, np.nan) for v in ok])) for k in keys}
    out["summary_min"] = {k: float(np.nanmin([v.get(k, np.nan) for v in ok])) for k in ("precision", "precision_hq", "coverage_hq")}
    cats = sorted({k for v in ok for k in v["abstention_hq"]})
    out["abstention_hq_median"] = {k: float(np.median([v["abstention_hq"].get(k, 0) for v in ok])) for k in cats}
    cats = sorted({k for v in ok for k in v["abstention_all"]})
    out["abstention_all_median"] = {k: float(np.median([v["abstention_all"].get(k, 0) for v in ok])) for k in cats}
    Path(f"reports/phase15/id_{split}_{tag}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(tag, {k: round(v, 3) for k, v in out["summary"].items()}, "min", {k: round(v, 3) for k, v in out["summary_min"].items()})
    print("  HQ abstention", {k: round(v, 3) for k, v in out["abstention_hq_median"].items()})
    print("  all abstention", {k: round(v, 3) for k, v in out["abstention_all_median"].items()})


if __name__ == "__main__":
    main()
