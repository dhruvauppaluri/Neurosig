"""Phase 11A: event-referencing strategies for real Neuropixels windows.

All rules below were fixed before any Phase 11 evaluation (see docs/phase11-localization.md):

A  sorter-referenced      corrected sorter times on the probe clock (sync_version >= 3), sorter peak channel.
B  detector-referenced    frozen detector-v2.0 float reference (reports/phase5/detector-v2.json, 200-4500 Hz, median
                          CAR, 4 sigma, negative) run on the raw window; event = merged trough sample/channel, shifted
                          by a constant delay calibrated once on development windows (median detector-sorter offset).
C  sorter + local align   corrected sorter time moved to the most negative *summed* filtered voltage over a fixed
                          contact set (contacts within 60 um of the sorter channel) within +-K samples (K=6, 0.2 ms).
                          One shift per event for all channels (no per-channel minima).

Matching A<->B: optimal one-to-one assignment (scripts.phase5.metrics.match_optimal), |dt| <= 0.5 ms, channel
consistent (detector channel within 60 um of the sorter channel). Chance = same matching against sorter times
circularly shifted by fixed offsets (Phase 5 convention).

Waveform quality is measured on the frozen real-detector preprocessing (300-6000 Hz causal, median CAR): SNR =
trough depth within +-0.2 ms on the best contact of the fixed set divided by that contact's MAD noise.
A "measurable trough" is SNR >= 4 (the detector threshold). Localization snippets are always cut from the raw
window, exactly as in Phases 9-10, so only the reference time differs between strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from edge.phase9.events import window_array, window_sorter_seconds
from edge.snippets import extract_snippet
from scripts.phase5.detectors import P5Config, run_detector
from scripts.phase5.metrics import match_optimal
from scripts.real_detector import frozen_real_detector, preprocess

ROOT = Path(__file__).resolve().parents[2]
MATCH_TOL_MS = 0.5
CHANNEL_RADIUS_UM = 60.0
ALIGN_HALF = 6                 # samples (0.2 ms at 30 kHz)
ALIGN_RADIUS_UM = 60.0
SNR_HALF = 6
TROUGH_SNR = 4.0
CHANCE_SHIFTS_S = (0.25, 0.5, 0.75)


def detector_v2_config() -> P5Config:
    cfg = json.loads((ROOT / "reports/phase5/detector-v2.json").read_text())["configs"]["v2"]
    return P5Config(**cfg)


@dataclass
class WindowEvents:
    window: dict
    raw: np.ndarray            # [T, C] raw AP uV
    align: np.ndarray          # [T, C] frozen preprocessing (300-6000 Hz causal, median CAR)
    sigma: np.ndarray          # [C] MAD noise of ``align``
    array: object
    fs: float
    sorter_sample: np.ndarray
    sorter_channel: np.ndarray
    sorter_cluster: np.ndarray
    det_sample: np.ndarray     # raw detector trough samples (no delay applied)
    det_channel: np.ndarray
    det_runtime_s: float


def _mad(x):
    return np.maximum(np.median(np.abs(x - np.median(x, axis=0)), axis=0) / 0.6745, 1e-9)


def load_events(w: dict, d: dict) -> WindowEvents:
    fs = float(d["fs"])
    raw = np.asarray(d["voltages_uv"], float)
    align, _ = preprocess(raw, frozen_real_detector(fs))
    t = window_sorter_seconds(d, "probe")
    s = np.rint(t * fs).astype(int)
    det = run_detector(raw, detector_v2_config(), coordinates_um=np.asarray(d["coordinates_um"]))
    ev = det["events"]
    return WindowEvents(w, raw, align, _mad(align), window_array(d), fs, s, np.asarray(d["sorter_local_channel"], int),
                        np.asarray(d["sorter_clusters"], int), np.array([e.sample for e in ev], int),
                        np.array([e.channel for e in ev], int), float(det["runtime_s"]))


def neighbours(array, ch: int, radius: float) -> np.ndarray:
    E = array.xyz_um()
    return np.flatnonzero(np.linalg.norm(E - E[ch], axis=1) <= radius)


def align_sample(we: WindowEvents, s: int, ch: int, half: int = ALIGN_HALF) -> int:
    """Strategy C: one shift for all contacts, chosen by the most negative summed filtered voltage over the fixed
    contact set around the reference channel."""
    S = neighbours(we.array, ch, ALIGN_RADIUS_UM)
    lo, hi = max(s - half, 0), min(s + half + 1, we.align.shape[0])
    if hi <= lo:
        return s
    score = (we.align[lo:hi][:, S] / we.sigma[S][None, :]).sum(axis=1)
    return lo + int(np.argmin(score))


def raw_car(we: WindowEvents) -> tuple[np.ndarray, np.ndarray]:
    """The signal the P9/P10 engines sample: raw AP minus the cross-channel median (static offsets do not affect the
    argmin of a short window). Cached on the WindowEvents object."""
    if getattr(we, "_rc", None) is None:
        rc = we.raw - np.median(we.raw, axis=1, keepdims=True)
        we._rc = (rc, _mad(rc))
    return we._rc


def align_sample_raw(we: WindowEvents, s: int, ch: int, half: int = 3) -> int:
    """Experiment 11A-E3: Strategy C rule applied to the engine's own signal (raw - median CAR), +-half samples."""
    rc, sig = raw_car(we)
    S = neighbours(we.array, ch, ALIGN_RADIUS_UM)
    lo, hi = max(s - half, 0), min(s + half + 1, rc.shape[0])
    if hi <= lo:
        return s
    base = np.median(rc[max(s - 40, 0):max(s - 12, 1)][:, S], axis=0)
    score = ((rc[lo:hi][:, S] - base[None, :]) / sig[S][None, :]).sum(axis=1)
    return lo + int(np.argmin(score))


def event_quality(we: WindowEvents, s: int, ch: int) -> dict:
    S = neighbours(we.array, ch, ALIGN_RADIUS_UM)
    lo, hi = max(s - SNR_HALF, 0), min(s + SNR_HALF + 1, we.align.shape[0])
    seg = we.align[lo:hi][:, S] / we.sigma[S][None, :]
    depth = -seg.min(axis=0)
    k = int(np.argmax(depth))
    return {"snr": float(depth[k]), "best_channel": int(S[k]), "trough": bool(depth[k] >= TROUGH_SNR),
            "trough_offset": int(lo + int(np.argmin(seg[:, k])) - s)}


def match_ab(we: WindowEvents, det_samples: np.ndarray, sorter_samples: np.ndarray | None = None,
             tol_ms: float = MATCH_TOL_MS) -> list[tuple[int, int, int]]:
    ref = we.sorter_sample if sorter_samples is None else sorter_samples
    if not len(det_samples) or not len(ref):
        return []
    E = we.array.xyz_um()
    allowed = np.linalg.norm(E[we.det_channel][:, None] - E[we.sorter_channel][None], axis=2) <= CHANNEL_RADIUS_UM
    return match_optimal(det_samples, ref, int(round(tol_ms * we.fs / 1000)), allowed)


def chance_matches(we: WindowEvents, det_samples: np.ndarray) -> float:
    n = we.raw.shape[0]
    out = []
    for sh in CHANCE_SHIFTS_S:
        shifted = (we.sorter_sample + int(round(sh * we.fs))) % n
        out.append(len(match_ab(we, det_samples, shifted)))
    return float(np.mean(out))


def snippet_at(we: WindowEvents, s: int, ch: int, event_id: int):
    if s - 40 < 0 or s + 60 >= we.raw.shape[0]:
        return None
    return extract_snippet(we.raw, int(s), we.array, event_id=int(event_id), sample_rate_hz=we.fs, peak_channel=int(ch))
