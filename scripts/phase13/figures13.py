"""Phase 13 figures (data/ibl-venv): (1) Section 11 simulation - cross-session decoding vs position error per spatial
code scale; (2) IBL per-spike depth-map decoding vs injected position noise (select + final sessions, T2 side);
(3) learned spatial representation: per depth bin, contralateral minus ipsilateral mean activity (z-scored per bin) of
the per-spike depth map, for each IBL session with T2 above its shuffled control. Writes reports/phase13/fig_*.png."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, ".")

R = Path("reports/phase13")
sim = json.loads((R / "sim13_localization_link.json").read_text())
fig, ax = plt.subplots(figsize=(6, 4))
sig = [0, 3, 5, 12, 25, 50, 100, 200]
for key, lab in (("M1_delta20", "20 um code"), ("M1_delta50", "50 um"), ("M1_delta100", "100 um"), ("M1_delta250", "250 um"), ("M2_delta50", "salt-and-pepper")):
    if key in sim:
        y = [sim[key][f"cross_unit_sigma{s}"]["mean"] for s in sig]
        ax.plot([max(s, 1) for s in sig], y, marker="o", label=lab)
ax.axvspan(2, 4, color="0.85", label="P12 unit-level precision (proxy)")
ax.axvline(12, color="0.5", ls="--", label="P12 single-event median deviation")
ax.axhline(0.25, color="k", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("unit position error sigma (um)"); ax.set_ylabel("cross-session accuracy (4 classes)")
ax.set_title("Simulation: when does localization precision matter?"); ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(R / "fig_sim_precision.png", dpi=130)

fig, ax = plt.subplots(figsize=(6, 4))
for split in ("select", "final"):
    p = R / f"ibl13_{split}.json"
    if not p.exists():
        continue
    S = json.loads(p.read_text())["summary"].get("T2_side")
    if not S:
        continue
    b = S["balanced_accuracy_mean"]
    xs = [1, 10, 25, 50, 100, 200]
    ys = [b["lr:smap"]] + [b[f"lr:smap_noise{s}"] for s in xs[1:]]
    ax.plot(xs, ys, marker="o", label=f"per-spike depth map ({split})")
    ax.axhline(b["lr:counts"], ls="--", lw=0.8, label=f"unit counts ({split})")
    ax.axhline(b["ctrl:shuffled_labels"], ls=":", lw=0.8, label=f"shuffled labels ({split})")
ax.set_xscale("log"); ax.set_xlabel("injected per-spike depth noise (um; 1 = none)"); ax.set_ylabel("balanced accuracy, contra vs ipsi")
ax.set_title("IBL: decoding from spatial maps vs position noise"); ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(R / "fig_ibl_noise.png", dpi=130)

# learned spatial representation (descriptive; no decoding)
from edge.phase13 import decode as D
from scripts.phase13.run_ibl13 import session, EDGE_UM
keys = [k for k in "EFGHIJKLMNOPQRS" if (Path("data/phase13") / f"ibl_{k}.npz").exists()]
fig, axes = plt.subplots(1, min(len(keys), 8), figsize=(2 * min(len(keys), 8), 4), sharey=False)
for ax, k in zip(np.atleast_1d(axes), keys[:8]):
    s = session(k)
    lo = np.floor(s["udep"].min() / EDGE_UM) * EDGE_UM; hi = np.ceil(s["udep"].max() / EDGE_UM + 1) * EDGE_UM
    edges = np.arange(lo, hi + 1, EDGE_UM)
    M = D.spike_depth_maps(s["per"], len(s["C"]), edges, 2, False)
    z = (M - M.mean(0)) / (M.std(0) + 1e-9)
    c, ip = s["contra"] > 0, s["ipsi"] > 0
    diff = z[c & ~ip].mean(0) - z[ip & ~c].mean(0)
    ax.barh((edges[:-1] + edges[1:]) / 2, diff, height=EDGE_UM * 0.9)
    ax.axvline(0, color="k", lw=0.6); ax.set_title(k, fontsize=9); ax.tick_params(labelsize=7)
np.atleast_1d(axes)[0].set_ylabel("probe depth (um)")
fig.suptitle("Per-spike depth maps: contralateral minus ipsilateral (z units), 40-120 ms", fontsize=9)
fig.tight_layout(); fig.savefig(R / "fig_ibl_spatial_maps.png", dpi=130)
print("figures written")
