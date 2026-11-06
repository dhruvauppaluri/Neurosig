"""Corrected re-run of the Phase 3/4 IBL detection + localization evaluations on the probe clock.

The ORIGINAL evaluation code (edge/ibl_eval.py: evaluate_window, edge/ibl.py: run_pilot) is run unchanged; only its raw-chunk
loader is replaced by one that serves cached windows with sorter times in either clock:
  control   : sorter times as the original code saw them (session clock, relative to the window start). Must reproduce the
              published numbers; this proves the clock offset accounts for the original behaviour.
  corrected : sorter times mapped session clock -> probe sample clock via _spikeglx_*.timestamps.npy.
No detector parameter is changed. Originals in reports/ibl-*.json are never modified.

    PYTHONPATH=. .venv/bin/python scripts/phase6/corrected_history.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

import edge.ibl as ibl
import edge.ibl_eval as ibl_eval
from scripts.phase5.data import load_window, window_path

OUT = ROOT / "reports" / "phase6" / "corrected_history"
ORIG = json.loads((ROOT / "reports" / "ibl-detection-localization.json").read_text())
PILOT = json.loads((ROOT / "reports" / "ibl-pilot.json").read_text())
LEGACY = [("4836a465-c691-4852-a0b1-dcd2b1ce38a1", 50.0), ("4836a465-c691-4852-a0b1-dcd2b1ce38a1", 100.0), ("4836a465-c691-4852-a0b1-dcd2b1ce38a1", 200.0),
          ("1a60a6e1-da99-4d4e-a734-39b1d4544fad", 50.0), ("1a60a6e1-da99-4d4e-a734-39b1d4544fad", 90.0)]


def fake_chunk(w: dict, clock: str) -> dict:
    sel = w["selected"]
    n_chan = int(sel.max()) + 1
    loc = np.full((n_chan, 2), np.nan)
    loc[sel] = w["coordinates_um"][:, :2]
    clusters = w["sorter_clusters"]
    cc = np.zeros(int(clusters.max()) + 1 if len(clusters) else 1, dtype=int)
    cc[clusters] = sel[w["sorter_local_channel"]]
    rel = w["sorter_times_s"] + (w["sync_offset_s"] if clock == "control" else 0.0)
    return {"one": None, "eid": "cached", "probe": "probe00", "fs": w["fs"], "locations": loc, "selected": sel,
            "array": ibl.probe_geometry(sel, loc), "raw_indices": sel, "voltages_uv": w["voltages_uv"],
            "spike_times": rel + w["t0_s"], "spike_clusters": clusters, "cluster_channels": cc,
            "sorter_mask": np.ones(len(rel), dtype=bool), "t0_s": w["t0_s"], "duration_s": w["duration_s"], "pid": w["pid"]}


def run_variant(clock: str) -> list[dict]:
    out = []
    for pid, t0 in LEGACY:
        p = window_path(pid, t0, 1.0)
        if not p.exists():
            out.append({"pid": pid, "t0_s": t0, "status": "window not available (download refused: HTTP 403)"})
            continue
        w = load_window(p)
        ibl_eval.load_raw_chunk = lambda *a, _w=w, **k: fake_chunk(_w, clock)
        r = ibl_eval.evaluate_window(t0, 1.0, pid=pid)
        r["sync_offset_ms_session_minus_probe"] = w["sync_offset_s"] * 1000
        out.append(r)
    return out


def pilot(clock: str) -> dict:
    w = load_window(window_path(PILOT["pid"], PILOT["t0_s"], PILOT["duration_s"]))
    ibl.load_raw_chunk = lambda *a, _w=w, **k: dict(fake_chunk(_w, clock), one=_FakeOne())
    return ibl.run_pilot(PILOT["pid"], PILOT["t0_s"], PILOT["duration_s"])


class _FakeOne:
    def load_dataset(self, *a, **k):
        import pandas as pd
        return pd.DataFrame(columns=PILOT["trial_columns"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"control": run_variant("control"), "corrected": run_variant("corrected")}
    try:
        res["pilot_control"], res["pilot_corrected"] = pilot("control"), pilot("corrected")
    except Exception as e:
        res["pilot_error"] = f"{type(e).__name__}: {e}"
    (OUT / "rerun_raw.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("written", OUT / "rerun_raw.json")


if __name__ == "__main__":
    main()
