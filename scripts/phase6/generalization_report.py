"""Baseline (300-6000 Hz) vs v2 (200-4500 Hz) vs v2+mean-CAR on identical windows, by dataset role and session.

Roles: development (calibration windows; also where ideas were first looked at), model_selection, previously_used_heldout (Phase 5
held-out windows, no longer a fresh holdout), seen_legacy (earlier-phase windows), locked_final_phase6 (new, evaluated once; only if
its windows could be downloaded). Same samples and same reference for every configuration; paired window-cluster bootstrap.
    .venv/bin/python scripts/phase6/generalization_report.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.phase5.benchmark import compare, evaluate_window, summarize
from scripts.phase5.data import load_window, manifest_windows, window_path
from scripts.phase5.detectors import BASELINE
from scripts.phase6.mismatch_analyzer import V2

CFGS = {"baseline": BASELINE, "v2": V2, "v2_meancar": V2.replace(car="mean")}
ROLES = {"development": ("calibration", "phase5"), "model_selection": ("selection", "phase5"), "previously_used_heldout": ("heldout", "phase5"),
         "seen_legacy": ("seen", "phase5"), "locked_final_phase6": ("locked_final", "phase6")}


def specs(split, which):
    if which == "phase5":
        return manifest_windows(split)
    p = ROOT / "reports" / "phase6" / "split_manifest_phase6.json"
    return [w for w in json.loads(p.read_text())["windows"] if w["split"] == split] if p.exists() else []


def main() -> None:
    out = {"roles": {}}
    for role, (split, which) in ROLES.items():
        items = []
        for sp in specs(split, which):
            p = window_path(sp["pid"], sp["t0_s"], sp["duration_s"])
            if p.exists():
                items.append((load_window(p), sp))
        n_expected = len(specs(split, which))
        if not items:
            out["roles"][role] = {"status": "no windows available", "expected": n_expected}
            continue
        res = {}
        for name, cfg in CFGS.items():
            caches = [dict() for _ in items]
            res[name] = [evaluate_window(w, sp, cfg, caches[i]) for i, (w, sp) in enumerate(items)]
        entry = {"status": "ran", "windows": len(items), "expected": n_expected, "sessions": sorted({sp["session"] for _, sp in items}), "configs": {}}
        for name in CFGS:
            s = summarize(res[name])
            entry["configs"][name] = {t: {m: {k: s["tol"][t][m][k] for k in ("f1", "recall", "precision", "chance_f1", "false_detections_per_s", "f1_ci95")} for m in ("channel_consistent", "unconstrained")} for t in ("0.5", "1.0", "2.0")}
            entry["configs"][name]["timing_1ms_channel_consistent"] = s["tol"]["1.0"]["timing_channel_consistent"]
            entry["configs"][name]["n_events"] = s["n_events"]; entry["configs"][name]["runtime_s_total"] = s["runtime_s_total"]
            entry["configs"][name]["per_session_f1_1ms"] = {k: v["1.0"]["channel_consistent"]["f1"] for k, v in s["per_session"].items()}
        entry["v2_vs_baseline"] = {}
        for tol in (0.5, 1.0, 2.0):
            c = compare(res["baseline"], res["v2"], tol); entry["v2_vs_baseline"][str(tol)] = c
        entry["v2_meancar_vs_v2_1ms"] = compare(res["v2"], res["v2_meancar"], 1.0)
        # window-level wins and per-session wins; per-channel recall
        a = [r["tol"]["1.0"]["channel_consistent"]["f1"] for r in res["v2"]]; b = [r["tol"]["1.0"]["channel_consistent"]["f1"] for r in res["baseline"]]
        entry["windows_v2_better"] = int(sum(x > y for x, y in zip(a, b))); entry["windows_v2_worse"] = int(sum(x < y for x, y in zip(a, b)))
        ch = {}
        for name in ("baseline", "v2"):
            agg = {}
            for r, (w, sp) in zip(res[name], items):
                for c, (h, n) in r["per_channel_recall_1ms"].items():
                    k = f"{sp['session']}:{c}"; agg.setdefault(k, [0, 0]); agg[k][0] += h; agg[k][1] += n
            ch[name] = agg
        keys = [k for k in ch["v2"] if ch["v2"][k][1] >= 20]
        rec = lambda name, k: ch[name][k][0] / ch[name][k][1]
        entry["per_channel_recall_1ms"] = {"channels_with_>=20_reference_spikes": len(keys),
                                           "v2_recall_ge_baseline": int(sum(rec("v2", k) >= rec("baseline", k) for k in keys)),
                                           "median_delta_recall": float(np.median([rec("v2", k) - rec("baseline", k) for k in keys])) if keys else None}
        out["roles"][role] = entry
        v = entry["v2_vs_baseline"]["1.0"]
        print(f"{role:26s} windows {len(items):2d}  dF1(v2-base) {v['delta_f1']:+.4f} CI [{v['delta_f1_ci95'][0]:+.4f},{v['delta_f1_ci95'][1]:+.4f}]  better in {entry['windows_v2_better']}/{len(items)}  per-session {{ {', '.join(f'{k}:{x:+.3f}' for k, x in v['delta_f1_per_session'].items())} }}", flush=True)
    (ROOT / "reports" / "phase6" / "generalization.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
