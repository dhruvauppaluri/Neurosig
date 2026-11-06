"""Phase 5 diagnostics of the frozen CAR/4σ detector on the CALIBRATION windows.

Run: .venv/bin/python scripts/phase5/diagnose_real.py [--split calibration]

Classifies every sorter-reference spike and every detection:
  detection failure      no candidate event exists near the reference spike
  timing failure         an event exists within 5 ms but outside 1 ms
  suppression            a candidate existed but merge / artifact veto / refractory removed it
  reference disagreement detection without a sorter match (NOT called a false positive)
Only the calibration split is used; selection/held-out windows are never read here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from scripts.filter_iir import IirConfig, design_sos
from scripts.phase5.data import load_window, manifest_windows, window_path
from scripts.phase5.detectors import BASELINE, run_detector
from scripts.phase5.metrics import match_optimal

OUT = ROOT / "reports" / "phase5"
FIG = OUT / "figures"


def trough_sigma(ref, base, sigma, sample, half):
    lo, hi = max(0, sample - half), min(ref.shape[0], sample + half + 1)
    z = (ref[lo:hi] - base) / sigma
    ch = int(np.argmin(z.min(axis=0)))
    return float(-z.min()), ch, int(lo + np.argmin(z[:, ch]))


def analyse_window(w: dict, spec: dict, make_figs: bool) -> dict:
    fs = w["fs"]
    x = w["voltages_uv"]
    ref_s = np.rint(w["sorter_times_s"] * fs).astype(int)
    ref_ch = w["sorter_local_channel"]
    keep = (ref_s >= 0) & (ref_s < x.shape[0])
    ref_s, ref_ch = ref_s[keep], ref_ch[keep]
    half = int(round(0.5e-3 * fs))
    tol1, tol2, tol5 = (int(round(m * fs / 1000)) for m in (1.0, 2.0, 5.0))

    base_run = run_detector(x, BASELINE, w["coordinates_um"])
    events = base_run["events"]
    cands = base_run["candidates"]
    ref = base_run["reference"]
    base, sigma = base_run["baseline"], base_run["sigma"]
    det_s = np.array([e.sample for e in events], dtype=int)
    cand_s = np.array([e.sample for e in cands], dtype=int)

    # same candidates without refractory / veto / merge to find who suppressed what
    norefr = run_detector(x, BASELINE.replace(refractory_ms=0.0, artifact_channel_fraction=2.0, merge_ms=0.0),
                          w["coordinates_um"])
    nr_s = np.array([e.sample for e in norefr["candidates"]], dtype=int)

    det_ch = np.array([e.channel for e in events], dtype=int)
    cand_ch = np.array([e.channel for e in cands], dtype=int)
    nr_ch = np.array([e.channel for e in norefr["candidates"]], dtype=int)
    pos = np.asarray(w["coordinates_um"])[:, :3]

    def chan_near(times, chans, s, c, tol, radius=60.0):
        if not len(times):
            return False
        m = np.abs(times - s) <= tol
        if not m.any():
            return False
        return bool(np.any(np.linalg.norm(pos[chans[m]] - pos[c], axis=1) <= radius))

    pairs1 = {r: (d, dt) for d, r, dt in match_optimal(det_s, ref_s, tol1)}
    pairs2 = {r: (d, dt) for d, r, dt in match_optimal(det_s, ref_s, tol2)}
    matched_det = {d for d, _ in pairs1.values()}

    # zero-phase reference trough to isolate causal-filter latency
    sos = design_sos(IirConfig(sample_rate_hz=fs))
    zp = signal.sosfiltfilt(sos, x, axis=0)
    zp -= np.median(zp, axis=1, keepdims=True)

    categories = Counter()
    miss_rows, offsets_causal, offsets_zp = [], [], []
    for i, (s, c) in enumerate(zip(ref_s, ref_ch)):
        depth, ch, tr = trough_sigma(ref, base, sigma, s, half)
        offsets_causal.append(tr - s)
        zlo, zhi = max(0, s - half), min(zp.shape[0], s + half + 1)
        offsets_zp.append(int(zlo + np.argmin(zp[zlo:zhi, c]) - s))
        if i in pairs1:
            categories["matched"] += 1
            continue
        if chan_near(det_s, det_ch, s, c, tol5) and not chan_near(det_s, det_ch, s, c, tol1):
            cat = "timing_failure_channel_consistent"
        elif chan_near(cand_s, cand_ch, s, c, tol1) and not chan_near(det_s, det_ch, s, c, tol1):
            cat = "suppressed_by_merge_or_veto"
        elif chan_near(nr_s, nr_ch, s, c, tol1):
            cat = "suppressed_by_refractory"
        elif depth >= 4.0:
            cat = "crossing_missed_peak_search"
        elif depth >= 3.0:
            cat = "detection_failure_3_to_4_sigma"
        else:
            cat = "detection_failure_below_3_sigma"
        categories[cat] += 1
        miss_rows.append(dict(ref_sample=int(s), ref_channel=int(c), category=cat, trough_sigma=depth,
                              nearest_det_ms=float(np.min(np.abs(det_s - s)) / fs * 1000) if len(det_s) else None))

    # unmatched detections
    fp_rows = []
    ref_ch_all = ref_ch
    near_ref = lambda t: np.min(np.abs(ref_s - t)) if len(ref_s) else 10**9
    n_ch = x.shape[1]
    sim = Counter()
    for k, e in enumerate(events):
        if k in matched_det:
            continue
        d = near_ref(e.sample)
        n_simul = int(np.sum(np.abs(cand_s - e.sample) <= int(round(0.2e-3 * fs))))
        row = dict(sample=int(e.sample), channel=int(e.channel), amp_sigma=float(e.normalized_amplitude),
                   nearest_ref_ms=float(d / fs * 1000), simultaneous_candidates=n_simul)
        row["kind"] = ("near_reference_same_region_timing" if (d <= tol5 and chan_near(ref_s, ref_ch_all, e.sample, e.channel, tol5)) else
                       "near_reference_other_region" if d <= tol5 else
                       "multi_channel_event" if n_simul >= 3 else
                       "isolated_strong" if e.normalized_amplitude >= 6 else "isolated_weak")
        sim[row["kind"]] += 1
        fp_rows.append(row)

    # chance level: same classification with the reference shifted by 41 ms
    # (a lag at which true coincidences vanish but event density is unchanged)
    shift = int(round(0.041 * fs))
    null_hit = sum(1 for sm, cm_ in zip(ref_s, ref_ch)
                   if chan_near(det_s, det_ch, sm + shift, cm_, tol5) and not chan_near(det_s, det_ch, sm + shift, cm_, tol1))
    null_match_1ms = len(match_optimal(det_s, ref_s + shift, tol1))
    dts = []
    for sm, cm_ in zip(ref_s, ref_ch):
        m = np.abs(det_s - sm) <= tol5
        for j in np.flatnonzero(m):
            if np.linalg.norm(pos[det_ch[j]] - pos[cm_]) <= 60.0:
                dts.append(int(det_s[j] - sm))
    dt_hist = np.histogram(np.array(dts) / fs * 1000, bins=np.arange(-5.0, 5.01, 0.2))[0].tolist()

    # ceiling imposed by global 0.5 ms merge: refs with another ref within 0.5 ms
    ms05 = int(round(0.5e-3 * fs))
    order = np.sort(ref_s)
    crowded = int(np.sum((np.diff(order) <= ms05))) if len(order) > 1 else 0
    same_ch_isi = 0
    for c in set(ref_ch.tolist()):
        t = np.sort(ref_s[ref_ch == c])
        same_ch_isi += int(np.sum(np.diff(t) < int(round(1e-3 * fs))))

    # common mode / noise
    raw_f = signal.sosfilt(sos, x, axis=0)
    cm = np.median(raw_f, axis=1)
    cm_power_frac = float(np.var(cm) / np.mean(np.var(raw_f, axis=0)))
    sig_ratio = sigma / np.median(sigma)
    per_channel_ref = Counter(ref_ch.tolist())
    pos_vs_neg = []
    for s in ref_s[:400]:
        lo, hi = max(0, s - half), min(ref.shape[0], s + half + 1)
        z = (ref[lo:hi] - base) / sigma
        pos_vs_neg.append(float(z.max() > -z.min()))

    res = dict(
        session=spec["session"], t0_s=spec["t0_s"], n_samples=int(x.shape[0]),
        n_reference=int(len(ref_s)), n_events=int(len(events)), n_candidates=int(len(cands)),
        reference_categories=dict(categories),
        null_shifted41ms=dict(timing_like_hits=int(null_hit), matches_1ms=int(null_match_1ms)),
        channel_consistent_dt_hist_ms_bins_minus5_to_5_step0p2=dt_hist, unmatched_detection_kinds=dict(sim),
        merge_ceiling_refs_within_0p5ms=crowded, same_channel_refs_isi_under_1ms=same_ch_isi,
        timing=dict(
            causal_trough_minus_sorter_samples_median=float(np.median(offsets_causal)),
            causal_trough_minus_sorter_samples_mad=float(np.median(np.abs(np.array(offsets_causal) - np.median(offsets_causal)))),
            zero_phase_trough_minus_sorter_samples_median=float(np.median(offsets_zp)),
            matched_signed_dt_samples_1ms=[int(v[1]) for v in pairs1.values()],
            matched_signed_dt_samples_2ms=[int(v[1]) for v in pairs2.values()],
        ),
        noise=dict(sigma_uv_median=float(np.median(sigma)), sigma_ratio_min=float(sig_ratio.min()),
                   sigma_ratio_max=float(sig_ratio.max()), noisy_channels_gt_2x=int(np.sum(sig_ratio > 2)),
                   quiet_channels_lt_0p5x=int(np.sum(sig_ratio < 0.5))),
        common_mode_power_fraction=cm_power_frac,
        fraction_refs_with_larger_positive_than_negative_peak=float(np.mean(pos_vs_neg)) if pos_vs_neg else None,
        refs_per_channel=dict(sorted(per_channel_ref.items())),
        missed_examples=miss_rows[:200], unmatched_detection_examples=fp_rows[:200],
    )
    if make_figs:
        figs = []
        want = {}
        for r in miss_rows:
            want.setdefault(r["category"], r)
        for r in fp_rows:
            want.setdefault("unmatched_" + r["kind"], r)
        for cat, r in want.items():
            s = r.get("ref_sample", r.get("sample"))
            c = r.get("ref_channel", r.get("channel"))
            lo, hi = max(0, s - 60), min(ref.shape[0], s + 90)
            t = (np.arange(lo, hi) - s) / fs * 1000
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t, (ref[lo:hi, c] - base[c]) / sigma[c], lw=1, label="CAR filtered (σ)")
            ax.plot(t, (zp[lo:hi, c] - base[c]) / sigma[c], lw=0.8, alpha=0.6, label="zero-phase (σ)")
            ax.axhline(-4, color="orange", ls="--", lw=0.8)
            for d in det_s[(det_s >= lo) & (det_s < hi)]:
                ax.axvline((d - s) / fs * 1000, color="r", alpha=0.6)
            for q in ref_s[(ref_s >= lo) & (ref_s < hi)]:
                ax.axvline((q - s) / fs * 1000, color="g", ls=":")
            ax.set(title=f"{spec['session']} t0={spec['t0_s']:.0f}s ch{c}: {cat}", xlabel="ms", ylabel="σ")
            ax.legend(fontsize=6)
            fig.tight_layout()
            name = f"fail_{spec['session']}_{int(spec['t0_s'])}_{cat}.png"
            fig.savefig(FIG / name, dpi=110)
            plt.close(fig)
            figs.append(name)
        res["figures"] = figs
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="calibration", choices=["calibration"])
    ap.add_argument("--out", type=Path, default=OUT / "diagnostics.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    results, missing = [], []
    figs_done = set()
    for spec in manifest_windows(args.split):
        p = window_path(spec["pid"], spec["t0_s"], spec["duration_s"])
        if not p.exists():
            missing.append(spec)
            continue
        w = load_window(p)
        results.append(analyse_window(w, spec, make_figs=spec["session"] not in figs_done))
        figs_done.add(spec["session"])
    agg = Counter()
    for r in results:
        agg.update(r["reference_categories"])
    unm = Counter()
    for r in results:
        unm.update(r["unmatched_detection_kinds"])
    all_dt = [d for r in results for d in r["timing"]["matched_signed_dt_samples_2ms"]]
    summary = dict(
        split=args.split, n_windows=len(results), windows_missing=len(missing),
        reference_categories_total=dict(agg), unmatched_detection_kinds_total=dict(unm),
        null_shifted41ms_total=dict(timing_like_hits=sum(r["null_shifted41ms"]["timing_like_hits"] for r in results),
                                    matches_1ms=sum(r["null_shifted41ms"]["matches_1ms"] for r in results)),
        channel_consistent_dt_hist_total=np.sum([r["channel_consistent_dt_hist_ms_bins_minus5_to_5_step0p2"] for r in results], axis=0).tolist() if results else None,
        total_reference=sum(r["n_reference"] for r in results), total_events=sum(r["n_events"] for r in results),
        matched_signed_dt_samples_2ms_median=float(np.median(all_dt)) if all_dt else None,
        causal_minus_sorter_median_samples=float(np.median([r["timing"]["causal_trough_minus_sorter_samples_median"] for r in results])) if results else None,
        zero_phase_minus_sorter_median_samples=float(np.median([r["timing"]["zero_phase_trough_minus_sorter_samples_median"] for r in results])) if results else None,
    )
    args.out.write_text(json.dumps(dict(summary=summary, windows=results), indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
