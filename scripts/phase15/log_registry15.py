"""Phase 15 experiment registry: one JSON line per experiment, compiled from the per-experiment reports in
reports/phase15/ (identity id_*, cross-animal decoding xa_*, overlap overlap_*, uncertainty, runtime, real count).

Every line: id, track, split, config, key metrics, report path and the decision recorded for that experiment.
Decisions are rules fixed before the final run: a variant is retained only if it improves its track's primary
metric on development AND on selection data without breaking the guard metric (identity: HQ precision; overlap:
nested-threshold false-two rate <= 2%; decoding: mean lift over chance across the six tasks).
Usage: .venv/bin/python scripts/phase15/log_registry15.py  -> reports/phase15/experiment_registry.jsonl
"""
import glob
import json
from pathlib import Path

R = Path("reports/phase15")
TASKS = ("SG_sf5", "SG_orientation6", "DG_orientation8", "GB_x", "GB_y", "NS_image118")
CH = dict(zip(TASKS, (0.2, 1 / 6, 0.125, 1 / 9, 1 / 9, 1 / 118)))
fr = json.loads((R / "frozen_p15.json").read_text()) if (R / "frozen_p15.json").exists() else {}
D_TAGS = set(fr.get("vision", {}).get("D-P15_tags", []))
rows = []


def add(**kw):
    rows.append(kw)


# Track D: cross-animal decoding
for f in sorted(glob.glob(str(R / "xa_*.json"))):
    d = json.loads(Path(f).read_text()); s = d.get("summary", {})
    if len(s) < 6:
        continue
    lift = sum(s[t] - CH[t] for t in TASKS) / 6
    add(id=f"D.{d['tag']}", track="D vision", split=d["mode"], config=d["spec"], metrics={**{t: round(s[t], 4) for t in TASKS}, "mean_lift": round(lift, 4)},
        report=Path(f).name, decision="retained (D-P15)" if d["tag"] in D_TAGS else ("final" if d["mode"] == "final" else "not retained"))

# Track C: identity
for f in sorted(glob.glob(str(R / "id_*.json"))):
    d = json.loads(Path(f).read_text()); s = d.get("summary", {})
    keep = {k: round(s[k], 4) for k in ("precision_hq", "coverage_hq", "coverage_all", "precision", "ari", "fragmentation_mean") if k in s}
    cfg = d.get("cfg", {})
    retained = cfg == fr.get("identity", {}).get("ID-P15")
    add(id=f"C.{Path(f).stem[3:]}", track="C identity", split=d.get("split"), config=cfg, metrics=keep, report=Path(f).name,
        decision="retained (ID-P15)" if retained else ("reference OU-P14" if cfg in ({}, None) else "not retained"))

# Track B: overlap / count
for f in sorted(glob.glob(str(R / "overlap_lomo*.json"))):
    d = json.loads(Path(f).read_text())
    for m, v in d.items():
        if isinstance(v, dict):
            add(id=f"B.{Path(f).stem}.{m}", track="B overlap", split="LOMO (13 morphologies)", config={"model": m, "nested": "nested" in f},
                metrics={k: v[k] for k in v if isinstance(v[k], (int, float))}, report=Path(f).name,
                decision="retained (B-P15)" if ("line20-ms3" in f and m == "hgb") else "comparison")

for name, track in (("uncertainty15_lomo.json", "U uncertainty"), ("batch_test.json", "F runtime"), ("tcompress_test.json", "F runtime"),
                    ("real_count15_select.json", "B overlap (real proxy)"), ("count_b_p15.json", "B overlap")):
    p = R / name
    if p.exists():
        d = json.loads(p.read_text())
        if name == "count_b_p15.json":
            d = {k: d[k] for k in ("features", "quantile", "nested_fp_by_quantile") if k in d}
        add(id=f"{track[0]}.{p.stem}", track=track, split="see report", config={}, metrics=d if len(json.dumps(d)) < 4000 else {"see": name},
            report=name, decision="retained" if name in ("uncertainty15_lomo.json", "batch_test.json", "count_b_p15.json") else "measurement")

p = R / "stream15.json"
if p.exists():
    for r in json.loads(p.read_text()):
        if r.get("mode") == "compute":
            add(id=f"F.compute.{r['session']}", track="F runtime", split="stream", config={"threads": r["threads"]}, metrics=r, report="stream15.json", decision="measurement")
            continue
        add(id=f"F.stream.{r['session']}.w{r['workers']}.x{r['speed']}", track="F runtime", split="stream", config={k: r[k] for k in ("workers", "speed", "batch")},
            metrics=r, report="stream15.json", decision="measurement")

# experiments without a result file of their own
add(id="D.e17_fclust_mlp", track="D vision", split="dev", config={"unit_norm": "none", "groups": ["area", "fclust"], "model": "mlp"},
    metrics={}, report=None, decision="not completed: stopped while the laptop was swapping (12 GB); not needed for selection")
add(id="C.online_split", track="C identity", split="dev/select", config={"split_every": 50, "split_dprime": [3, 4, 5]}, metrics={"see": "id_dev_s_*.json"},
    report="id_dev_s_tau_rel0.2split_every50split_dprime4.json", decision="rejected: no dev gain; crashed on select (shape mismatch), not fixed")
add(id="F.stream_overload", track="F runtime", split="stream", config={"speed": 50}, metrics={"see": "stream15_overload.json"},
    report="stream15_overload.json", decision="measurement of drop behaviour under 8,000-19,000 events/s offered; not capacity")
p = R / "xa_final_comparison.json"
if p.exists():
    d = json.loads(p.read_text())
    add(id="D.final_comparison", track="D vision", split="final", config={"frozen_hash": d["frozen_hash"]},
        metrics={t: {k: r[k] for k in ("D-P15", "reference_p14_spec")} | {"diff": r["D-P15_minus_reference"]["mean_diff"], "ci95": r["D-P15_minus_reference"]["ci95"],
                 "sign_p": r["D-P15_minus_reference"]["sign_test_p"]} for t, r in d["tasks"].items()}, report="xa_final_comparison.json", decision="final (single use)")

with open(R / "experiment_registry.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, default=str) + "\n")
print(len(rows), "registry lines")
