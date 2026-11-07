"""Calibrate on dev, check coverage/risk on select."""
import numpy as np
from edge.benchmark_v3 import _array
from edge.phase8.pipeline import Phase8Localizer, fit_calibration, save_calibration
from edge.phase8.synth import make_events

array = _array("linear")
dev = make_events("dev", n_per_cell=150)
sel = make_events("select", n_per_cell=150)
cal = fit_calibration(array, dev)
print({k: round(v, 2) if isinstance(v, float) else v for k, v in cal.to_json().items()}, cal.__dict__.get("_n_scores"))
save_calibration(cal, "reports/phase8/calibration_v2.json")
loc = Phase8Localizer(array, cal)
rows = []
for e in sel:
    r = loc.localize(e.snippet)
    p = r.primary
    truth = np.array([np.hypot(e.truth_xyz_um[0], e.truth_xyz_um[2]), e.truth_xyz_um[1]])
    if p is None:
        rows.append((e.noise_uv, e.contaminated, r.accepted, np.nan, np.nan, r.n_sources, np.nan)); continue
    d = np.array(p.xyz_um[:2]) - truth
    m = d @ np.linalg.solve(np.array(p.cov_xy), d)
    key_q = cal.q_shared if 'shared_waveform_pair' in r.flags else (cal.q_two if r.n_sources == 2 else (cal.q_flagged if r.flags else cal.q_clean))
    rows.append((e.noise_uv, e.contaminated, r.accepted, np.linalg.norm(d), float(m <= key_q), r.n_sources, p.region_area_um2))
R = np.array([x + (np.nan,) * (7 - len(x)) for x in rows], float)
acc = R[:, 2] == 1
print("accepted", acc.mean().round(3))
for name, mk in {"all": np.ones(len(R), bool), "single": R[:, 1] == 0, "contam": R[:, 1] == 1,
                 "n2": R[:, 0] == 25.0}.items():
    for lab, m in (("all", mk), ("accepted", mk & acc)):
        e = R[m, 3]
        print(f"{name:7s} {lab:8s} n={m.sum():3d} med={np.nanmedian(e):5.1f} p90={np.nanpercentile(e,90):5.1f} "
              f"p95={np.nanpercentile(e,95):5.1f} coverage={np.nanmean(R[m,4]):.2f} area_med={np.nanmedian(R[m,6]):.0f}")
two_pred = R[:, 5] == 2
print("false2 single:", two_pred[R[:, 1] == 0].mean().round(3), " detect2 contam:", two_pred[R[:, 1] == 1].mean().round(3))
