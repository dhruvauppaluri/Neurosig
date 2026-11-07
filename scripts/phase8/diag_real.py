import json, sys
import numpy as np
sys.path.insert(0, ".")
from scripts.phase8.real_stability import *
w = [x for x in windows_for("dev")][3]
path = window_path(w["pid"], w["t0_s"], w["duration_s"])
array, snips, meta, win = events_from_window(path, 1)
E = array.xyz_um()
print("array x unique", np.unique(E[:, 0]), "y range", E[:, 1].min(), E[:, 1].max())
p8 = Phase8Localizer(array, model="line_source", max_x_um=90.0)
f = p8.fitter
print("fitter x bounds", f.lo, f.hi, "planar", f.planar)
eng = baseline_engines(array)
bad = 0
for sn in snips[:40]:
    fit = f.fit(sn)
    s = fit.sources1[0]
    b = eng["beamform"].estimate(sn).xyz_um
    pk = int(np.argmin(sn.voltages_uv[sn.n_pre]))
    d = np.hypot(s.x - b[0], s.y - b[1])
    if d > 60:
        bad += 1
        print(f"p8=({s.x:.0f},{s.y:.0f}) beam=({b[0]:.0f},{b[1]:.0f}) peakch y={E[pk,1]:.0f} x={E[pk,0]:.0f} "
              f"amp={-sn.voltages_uv[sn.n_pre].min():.0f}uV delta={fit.delta:.0f}")
print("bad", bad, "of 40")
