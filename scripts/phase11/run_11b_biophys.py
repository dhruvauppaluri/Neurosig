"""Phase 11B independent check on MEArec/NEURON biophysical templates (known soma position; generated without any
NeuroSig forward model). Empirical models B/D are learned from REAL dev events only, so the biophysical set tests
generalization, not self-consistency. Probe-plane mapping: lateral = MEArec y, axial = MEArec z, off-plane = x.
For each template: the 32 contacts nearest the soma axially, the trough-sample amplitude vector, Gaussian noise at
peak SNR 20/8/4 (10 realizations, seeded). Error = in-plane distance to the soma; held-out contact error noise-free."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
import MEArec as mr
from edge.phase11 import forward_models as F
from edge.phase11.footprint import learn_models

T = mr.load_templates("data/phase11/biophys/templates.h5")
import MEAutility as mu
P = mu.return_mea("Neuropixels-128").positions                      # (x off-plane=0, y lateral, z axial)
E_all = np.stack([P[:, 1], P[:, 2], np.zeros(len(P))], axis=1)
evs = pickle.load(open("data/phase11/footprint_dev.pkl", "rb"))
rng = np.random.default_rng(3)
mb, md = learn_models([evs[i] for i in rng.choice(len(evs), min(2000, len(evs)), replace=False)], cache={})
models = {"line20": F.Model("line20"), "linefree": F.Model("linefree"), "radial": mb, "aniso": md,
          "twoline": F.Model("twoline", n_src=2), "line20c": F.Model("line20c", n_src=2)}
rows = []
for k in range(T.templates.shape[0]):
    loc = T.locations[k]; tmpl = np.asarray(T.templates[k], float)            # [C, t]
    sel = np.argsort(np.abs(E_all[:, 1] - loc[2]))[:32]
    E = E_all[sel]; W = tmpl[sel]
    t0 = int(np.unravel_index(np.argmin(W), W.shape)[1]); a0 = -W[:, t0]
    truth = np.array([loc[1], loc[2]])
    for snr in (20.0, 8.0, 4.0):
        sigma = np.full(32, a0.max() / snr)
        for rep in range(10):
            a = a0 + rng.normal(0, sigma)
            for n, m in models.items():
                f = F.fit(m, a, sigma, E, x_range=(-60.0, 60.0))
                th = f.theta[2 * int(np.argmax(f.amps)):2 * int(np.argmax(f.amps)) + 2] if n == "twoline" else f.theta[:2]
                rows.append({"template": k, "cell": str(T.celltypes[k]), "dist_um": float(loc[0]), "snr": snr, "model": n,
                             "err_um": float(np.linalg.norm(th - truth)), "axial_err_um": float(abs(th[1] - truth[1]))})
    for n, m in models.items():       # noise-free held-out contacts
        msk = np.ones(32, bool); msk[np.random.default_rng(k).choice(32, 8, replace=False)] = False
        s = np.full(32, a0.max() / 50); f = F.fit(m, a0, s, E, mask=msk, x_range=(-60.0, 60.0)); pr = F.predict(m, f, E)
        rows.append({"template": k, "model": n, "heldout_rel_err": float(np.sum((a0[~msk] - pr[~msk]) ** 2) / np.sum(a0[~msk] ** 2))})
out = {"n_templates": int(T.templates.shape[0]), "cells": sorted(set(map(str, T.celltypes))), "models": {}}
for n in models:
    o = {}
    for snr in (20.0, 8.0, 4.0):
        e = np.array([r["err_um"] for r in rows if r.get("model") == n and r.get("snr") == snr])
        ax = np.array([r["axial_err_um"] for r in rows if r.get("model") == n and r.get("snr") == snr])
        o[f"snr{int(snr)}"] = {"err_median": float(np.median(e)), "err_p90": float(np.percentile(e, 90)),
                               "axial_median": float(np.median(ax)), "axial_p90": float(np.percentile(ax, 90))}
    h = [r["heldout_rel_err"] for r in rows if r.get("model") == n and "heldout_rel_err" in r]
    o["heldout_contact_rel_err_median_noise_free"] = float(np.median(h))
    out["models"][n] = o
    print(n, json.dumps(o), flush=True)
Path("reports/phase11/exp11b_biophys.json").write_text(json.dumps(out, indent=1) + "\n")
