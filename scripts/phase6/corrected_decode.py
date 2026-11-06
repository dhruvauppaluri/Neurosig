"""Corrected re-run of the Phase 4 detector-event decode on the right timebase (needs data/ibl-venv and raw 40 s blocks).

The original `_session_block` mixed clocks: detector events live on the probe sample clock, `stimOn_times` on the session clock.
Here trial onsets are mapped session -> probe for the detector counts and block selection, while sorted counts stay entirely on
the session clock (spike times and onsets are both session clock). The original experiment code (`run`) is used unchanged.
    PYTHONPATH=. data/ibl-venv/bin/python scripts/phase6/corrected_decode.py
"""
import json, socket, sys
from pathlib import Path
socket.setdefaulttimeout(120)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from one.api import ONE

import edge.decode.detector_experiment as de
from edge.ibl import SORTING_REVISION, TRIALS_REVISION
from scripts.phase5.data import fetch_window, load_window, make_clock_maps
from scripts.real_detector import detect_real, frozen_real_detector

OUT = ROOT / "reports" / "phase6" / "corrected_history" / "ibl-detector-decode.corrected.json"


def corrected_block(pid, t0_s, duration_s):
    path = fetch_window(pid, t0_s, duration_s)
    w = load_window(path)
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international", silent=True, cache_dir="data/neuropixels/ibl_cache")
    eid, probe = one.pid2eid(pid)
    trials = one.load_dataset(eid, "_ibl_trials.table.pqt", collection="alf", revision=TRIALS_REVISION)
    onsets_session, labels = de._eligible_trials(trials)
    ts = np.asarray(one.load_dataset(eid, "_spikeglx_*.timestamps.npy", collection=f"raw_ephys_data/{probe}"), dtype=float)
    _, reverse = make_clock_maps(ts)
    onsets_probe = np.asarray(reverse(onsets_session), dtype=float) / w["fs"]
    keep = (onsets_probe >= t0_s + 0.05) & (onsets_probe < t0_s + duration_s - 0.25)
    onsets_probe, onsets_session, labels = onsets_probe[keep], onsets_session[keep], labels[keep]
    events = detect_real(w["voltages_uv"], frozen_real_detector(w["fs"]))["events"]
    det_counts = de._counts_from_events(np.array([e.sample for e in events], dtype=float), np.array([e.channel for e in events], dtype=int),
                                        onsets_probe, t0_s=t0_s, fs=w["fs"], n_channels=w["voltages_uv"].shape[1])
    col = f"alf/{probe}/pykilosort"
    st = np.asarray(one.load_dataset(eid, "spikes.times.npy", collection=col, revision=SORTING_REVISION), dtype=float)
    sc = np.asarray(one.load_dataset(eid, "spikes.clusters.npy", collection=col, revision=SORTING_REVISION), dtype=int)
    selected = np.unique(w["sorter_clusters"])[:32]
    if len(selected) < 32:
        selected = np.unique(sc)[:32]
    sorted_counts = de._sorted_counts(st, sc, selected, onsets_session)
    return {"pid": pid, "eid": str(eid), "t0_s": t0_s, "duration_s": duration_s, "n_trials": int(len(labels)), "n_detector_events": int(len(events)),
            "labels": labels, "detector_counts": det_counts, "sorted_counts": sorted_counts[:, : det_counts.shape[1]],
            "majority": float(max(np.bincount(labels, minlength=2)) / max(len(labels), 1)), "detector": de.FROZEN_REAL_DETECTOR_NAME,
            "clock_note": "trial onsets mapped session->probe for detector counts; sorted counts on the session clock"}


if __name__ == "__main__":
    de._session_block = corrected_block
    rep = de.run()
    rep["corrected"] = "probe-clock trial onsets; originals in reports/ibl-detector-decode.json unchanged"
    OUT.write_text(json.dumps(rep, indent=1, default=de._json_default) + "\n")
    print(json.dumps({k: rep[k] for k in ("primary", "within_session", "cross_recording")}, indent=1, default=de._json_default)[:2500])
