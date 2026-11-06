"""Deterministic RTL test-vector generator for fixed-v1.1 (detector-v2.0, 200-4500 Hz, median/mean CAR).

    .venv/bin/python scripts/phase6/generate_rtl_vectors.py [--out tests/vectors/phase6] [--check]

Every scenario is produced twice: by the vectorised model (scripts/phase5/fixed_detector.py) and by the independent
oracle (scripts/phase6/oracles.py). The generator aborts if they disagree, so a vector is only written when two separate
implementations agree. Files are plain text for $readmemh / testbench use:
  *.hex   one two's-complement hex word per line (adc 16-bit, everything else 24-bit)
  *.csv   events: timestamp,channel,code,delta,amp_q8 (signed decimal)
  manifest.json  SHA-256 of every file; --check regenerates in memory and compares.
Frames are sample-major, channel-minor (all channels of frame 0, then frame 1 ...). Dropped frames (valid=0) are absent
from the intermediate files; timestamps.txt lists the u32 frame counter of each present frame.
No RTL exists for this model; the vectors are an input to a future RTL phase.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.fixed_point_model import FixedPointConfig, fixed_iir_filter
from scripts.phase5.fixed_detector import FixedSpec, V2_SPEC, SIG_MAX, SIG_MIN, calibrate, car_stage, detect_channel, merge_events, run_fixed, spatial_veto, FEvent
from scripts.phase6 import oracles as O

H16 = lambda v: f"{int(v) & 0xFFFF:04x}"
H24 = lambda v: f"{int(v) & 0xFFFFFF:06x}"


def synth(seed, n, ch, noise=4.0, n_spikes=8, amp=-70, cm=0.0):
    r = np.random.default_rng(seed)
    x = r.normal(0, noise, (n, ch)) + (r.normal(0, cm, (n, 1)) if cm else 0)
    for s in np.linspace(200, n - 120, n_spikes).astype(int):
        c = int(r.integers(0, ch)); t = np.arange(-6, 12)
        x[s + t, c] += amp * np.exp(-0.5 * (t / 2.0) ** 2)
    return np.clip(np.rint(x), -512, 511).astype(np.int64)


def system_scenario(name, codes, spec, active=None, valid=None, note=""):
    n, ch = codes.shape
    active = np.ones(ch, bool) if active is None else np.asarray(active, bool)
    valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    r = run_fixed(codes, spec, active=active, valid=valid)
    sd = {"guard_bits": spec.guard_bits, "car": spec.car, "threshold_mult_q8": spec.threshold_mult_q8,
          "mad_iterations": spec.mad_iterations, "peak_search": spec.peak_search, "refractory": spec.refractory,
          "veto_window": spec.veto_window, "veto_fraction_q8": spec.veto_fraction_q8, "merge_window": spec.merge_window}
    sos = [tuple(int(v) for v in row) for row in r["sos"]]
    o_events, o_cands = O.pipeline(codes.tolist(), sos, active.tolist(), valid.tolist(), sd)
    key = lambda e: (e.sample, e.channel, e.code, e.delta, e.amp_q8)
    assert [key(e) for e in r["candidates"]] == o_cands, f"{name}: candidates differ from oracle"
    assert [key(e) for e in r["events"]] == o_events, f"{name}: events differ from oracle"
    pres = codes[valid]
    x = np.clip(pres.astype(np.int64) << spec.guard_bits, SIG_MIN, SIG_MAX)
    cfg = FixedPointConfig(signal_bits=24)
    s1, _ = fixed_iir_filter(x, r["sos"][0:1], cfg)
    s2, _ = fixed_iir_filter(s1.astype(np.int64), r["sos"][1:2], cfg)
    assert np.array_equal(s2, r["filtered"]), f"{name}: staged filter differs"
    for c in range(ch):  # oracle section outputs for this channel
        o1, o2 = O.biquad_cascade(pres[:, c].tolist(), sos, spec.guard_bits)
        assert o1 == s1[:, c].tolist() and o2 == s2[:, c].tolist(), f"{name}: filter ch{c} differs from oracle"
    filt = r["filtered"]
    ref = np.array([O.car_ref(filt[t].tolist(), active.tolist(), spec.car) for t in range(filt.shape[0])]) if spec.car != "none" else np.zeros(filt.shape[0], int)
    car = r["car"]
    assert np.array_equal(car, np.array([[O.sat24(int(filt[t, c]) - int(ref[t])) for c in range(ch)] for t in range(filt.shape[0])]))
    cal = r["calibration"]
    files = {
        "adc_codes.hex": [H16(v) for v in pres.reshape(-1)],
        "timestamps.txt": [str(int(t)) for t in r["timestamps"]],
        "section1_out.hex": [H24(v) for v in s1.reshape(-1)],
        "section2_out.hex": [H24(v) for v in s2.reshape(-1)],
        "car_ref.hex": [H24(v) for v in ref],
        "car_out.hex": [H24(v) for v in car.reshape(-1)],
        "candidates.csv": ["timestamp,channel,code,delta,amp_q8"] + [f"{e.sample},{e.channel},{e.code},{e.delta},{e.amp_q8}" for e in r["candidates"]],
        "events.csv": ["timestamp,channel,code,delta,amp_q8"] + [f"{e.sample},{e.channel},{e.code},{e.delta},{e.amp_q8}" for e in r["events"]],
    }
    # candidates surviving the veto, before merge
    kept, nveto = spatial_veto(r["candidates"], int(active.sum()), spec)
    files["candidates_after_veto.csv"] = ["timestamp,channel,code,delta,amp_q8"] + [f"{e.sample},{e.channel},{e.code},{e.delta},{e.amp_q8}" for e in kept]
    meta = {"scenario": name, "note": note, "n_frames_total": int(n), "n_frames_present": int(valid.sum()), "channels": int(ch),
            "active": active.astype(int).tolist(), "spec": spec.as_dict(), "sos_q30_rows_b0_b1_b2_a0_a1_a2": [list(map(int, row)) for row in r["sos"]],
            "calibration": {k: np.asarray(v).astype(int).tolist() for k, v in cal.items()},
            "counts": {"candidates": len(r["candidates"]), "after_veto": len(kept), "vetoed": nveto, "events": len(r["events"])},
            "valid_mask": valid.astype(int).tolist() if not valid.all() else "all"}
    files["config.json"] = json.dumps(meta, indent=1).split("\n")
    return files


def unit_scenarios():
    out = {}
    # --- detector unit vectors (CAR-domain s24 traces)
    rng = np.random.default_rng(6001)
    cases = []
    base = [10] * 6 + [-5] + [10] * 80
    cases.append(("equality_and_crossing", np.array([10, 10, 5, 3, 4, 10, 10, 10, 5, 5, 6, 10] + [10] * 60), 5, 10))
    tr = np.full(200, 10); tr[100] = 0; tr[130] = 0
    cases.append(("refractory_exact", tr, 5, 10))
    tr = np.full(200, 10); tr[100] = 0; tr[129] = 0
    cases.append(("refractory_minus1", tr, 5, 10))
    tr = np.full(60, 10); tr[0], tr[1] = 10, 0
    cases.append(("stream_start_crossing_at_1", tr, 5, 10))
    tr = np.full(60, 10); tr[57] = 0; tr[59] = -3
    cases.append(("stream_end_truncated_search", tr, 5, 10))
    tr = np.full(80, 10); tr[20:24] = [2, 2, 2, 2]
    cases.append(("tied_minima_first_wins", tr, 5, 10))
    cases.append(("random_small_alphabet", rng.integers(-6, 7, 400), -2, 1))
    lines = []
    for name, trace, thr, bs in cases:
        trace = np.asarray(trace, dtype=np.int64); ts = np.arange(len(trace))
        recip = 3_000_000_000 // 7
        got = detect_channel(trace, ts, thr, bs, recip, 0, V2_SPEC)
        want = O.detect(trace.tolist(), ts.tolist(), thr, bs, recip, V2_SPEC.peak_search, V2_SPEC.refractory)
        assert [(e.sample, e.code, e.delta, e.amp_q8) for e in got] == want, name
        out[f"detector_unit/{name}.trace.hex"] = [H24(v) for v in trace]
        out[f"detector_unit/{name}.expected.csv"] = ["timestamp,code,delta,amp_q8"] + [",".join(map(str, w)) for w in want]
        out[f"detector_unit/{name}.params.json"] = json.dumps({"threshold": thr, "baseline": bs, "recip": recip, "peak_search": V2_SPEC.peak_search, "refractory": V2_SPEC.refractory}).split("\n")
    # --- dropped-frame detector vector
    trace = rng.integers(-6, 7, 300).astype(np.int64); valid = rng.random(300) > 0.08; valid[0] = True
    ts = np.arange(300)[valid]; tr = trace[valid]
    got = detect_channel(tr, ts, -3, 0, 12345678, 0, V2_SPEC); want = O.detect(tr.tolist(), ts.tolist(), -3, 0, 12345678, 15, 30)
    assert [(e.sample, e.code, e.delta, e.amp_q8) for e in got] == want
    out["detector_unit/dropped_frames.trace.hex"] = [H24(v) for v in tr]
    out["detector_unit/dropped_frames.timestamps.txt"] = [str(int(v)) for v in ts]
    out["detector_unit/dropped_frames.expected.csv"] = ["timestamp,code,delta,amp_q8"] + [",".join(map(str, w)) for w in want]
    # --- CAR unit vectors
    frames, exp = [], []
    for n_ch in (1, 2, 3, 4, 5, 8, 31, 32):
        for k in range(6):
            fr = rng.integers(-2 ** 23, 2 ** 23, n_ch)
            if k == 0: fr[:] = SIG_MAX
            if k == 1: fr[:] = SIG_MIN
            if k == 2: fr = np.array([(-3, -2, -1, 0)[i % 4] for i in range(n_ch)])
            frames.append(fr.tolist())
            exp.append([O.car_ref(fr.tolist(), [True] * n_ch, "median"), O.car_ref(fr.tolist(), [True] * n_ch, "mean")])
            got = car_stage(np.asarray(fr, dtype=np.int64)[None, :], np.ones(n_ch, bool), "median")[0]
            assert got.tolist() == [O.sat24(int(v) - exp[-1][0]) for v in fr]
    out["car_unit/frames.txt"] = [f"{len(f)} " + " ".join(H24(v) for v in f) for f in frames]
    out["car_unit/expected_ref_median_mean.csv"] = ["median_ref,mean_ref"] + [f"{a},{b}" for a, b in exp]
    # --- calibration unit vectors
    series = {"gauss": np.rint(rng.normal(0, 4000, 4000)), "outliers": np.concatenate([rng.normal(0, 2000, 3900), rng.normal(0, 400000, 100)]).round(),
              "mad_tie_2048": np.array([2048, -2048] * 400), "mad_zero": np.zeros(400), "tiny": rng.integers(-1, 2, 1000), "offset": np.rint(rng.normal(120000, 3000, 3000))}
    regs = ["name,baseline,mad,sigma_q16,threshold,recip,armed"]
    for nm, x in series.items():
        x = np.asarray(x, dtype=np.int64)
        w = O.calibrate(x.tolist())
        g = calibrate(x[:, None], np.array([True]), V2_SPEC)
        assert (int(g["baseline"][0]), int(g["mad"][0]), int(g["threshold"][0]), int(g["recip_q"][0])) == (w["baseline"], w["mad"], w["thr"], w["recip"]), nm
        out[f"calibration_unit/{nm}.car_out.hex"] = [H24(v) for v in x]
        regs.append(f"{nm},{w['baseline']},{w['mad']},{w['sigma_q16']},{w['thr']},{w['recip']},{int(w['armed'])}")
    out["calibration_unit/expected_registers.csv"] = regs
    # --- veto / merge unit vectors
    evs = [FEvent(int(rng.integers(0, 300)), int(rng.integers(0, 32)), -int(rng.integers(1, 99)), int(rng.integers(1, 99)), int(rng.integers(0, 5))) for _ in range(150)]
    evs = list({(e.sample, e.channel): e for e in evs}.values())
    tup = [(e.sample, e.channel, e.code, e.delta, e.amp_q8) for e in evs]
    kept, _ = spatial_veto(evs, 32, V2_SPEC); merged = merge_events(kept, V2_SPEC)
    assert [(e.sample, e.channel, e.code, e.delta, e.amp_q8) for e in merged] == O.merge(O.veto(tup, 32))
    out["merge_unit/candidates.csv"] = ["timestamp,channel,code,delta,amp_q8"] + [",".join(map(str, t)) for t in sorted(tup)]
    out["merge_unit/after_veto.csv"] = ["timestamp,channel,code,delta,amp_q8"] + [f"{e.sample},{e.channel},{e.code},{e.delta},{e.amp_q8}" for e in kept]
    out["merge_unit/events.csv"] = ["timestamp,channel,code,delta,amp_q8"] + [f"{e.sample},{e.channel},{e.code},{e.delta},{e.amp_q8}" for e in merged]
    return out


def all_files() -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    v2 = V2_SPEC
    scen = {
        "nominal_32ch_median": (synth(11, 1200, 32, n_spikes=10), v2, None, None, "32 channels, 10 injected spikes, median CAR"),
        "nominal_8ch_mean_car": (synth(12, 2000, 8), FixedSpec(highpass_hz=200.0, lowpass_hz=4500.0, car="mean"), None, None, "mean CAR variant (separate, not the frozen reference)"),
        "zero_input_4ch": (np.zeros((800, 4), dtype=np.int64), v2, None, None, "all channels disarmed (MAD = 0), no events"),
        "extremes_4ch": (np.concatenate([np.full((300, 4), 511), np.full((300, 4), -512), np.where(np.arange(400)[:, None] % 2 == 0, 511, -512) * np.ones((1, 4), dtype=np.int64)]).astype(np.int64), v2, None, None, "max/min native codes and Nyquist alternation"),
        "adc_container_extremes_2ch": (np.concatenate([np.full((200, 2), 32767), np.full((200, 2), -32768), np.where(np.arange(300)[:, None] % 2 == 0, 32767, -32768) * np.ones((1, 2), dtype=np.int64)]).astype(np.int64), v2, None, None, "s16 container extremes: shifted input saturates to s24"),
        "common_mode_8ch": (synth(14, 2000, 8, cm=0.0) + np.rint(150 * np.sin(2 * np.pi * 800 * np.arange(2000) / 30000))[:, None].astype(np.int64), v2, None, None, "large shared 800 Hz interference (clipped to native range)"),
    }
    cm = scen["common_mode_8ch"]; scen["common_mode_8ch"] = (np.clip(cm[0], -512, 511), *cm[1:])
    rng = np.random.default_rng(15)
    valid = rng.random(1500) > 0.02; valid[:3] = True
    act = np.array([1, 1, 0, 1, 1, 0, 1, 1], bool)
    scen["dropped_frames_disabled_channels_8ch"] = (synth(15, 1500, 8, n_spikes=8), v2, act, valid, "2 % dropped frames, channels 2 and 5 disabled")
    scen["overlapping_spikes_4ch"] = (np.clip(synth(16, 1500, 4, n_spikes=0) + _overlap(1500, 4), -512, 511), v2, None, None, "two spikes 12 samples apart on one channel plus simultaneous spikes on distant channels")
    for name, (codes, spec, active, valid_, note) in scen.items():
        for fn, lines in system_scenario(name, codes, spec, active, valid_, note).items():
            files[f"{name}/{fn}"] = lines
    files.update(unit_scenarios())
    return files


def _overlap(n, ch):
    x = np.zeros((n, ch), dtype=np.int64)
    t = np.arange(-6, 12)
    shape = np.rint(-80 * np.exp(-0.5 * (t / 2.0) ** 2)).astype(np.int64)
    x[500 + t, 0] += shape; x[512 + t, 0] += shape
    x[900 + t, 1] += shape; x[900 + t, 3] += np.rint(shape * 1.2).astype(np.int64)
    return x


def digest(lines): return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "tests" / "vectors" / "phase6")
    ap.add_argument("--check", action="store_true", help="regenerate in memory and compare with the committed manifest")
    a = ap.parse_args()
    files = all_files()
    manifest = {"model": "fixed-v1.1 / detector-v2.0", "files": {k: digest(v) for k, v in sorted(files.items())}}
    if a.check:
        cur = json.loads((a.out / "manifest.json").read_text())
        bad = [k for k in set(cur["files"]) | set(manifest["files"]) if cur["files"].get(k) != manifest["files"].get(k)]
        print("vector manifest matches" if not bad else f"MISMATCH in {sorted(bad)[:10]}")
        sys.exit(1 if bad else 0)
    for k, v in files.items():
        p = a.out / k; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("\n".join(v) + "\n")
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(len(files), "files,", sum(len(v) for v in files.values()), "lines ->", a.out)


if __name__ == "__main__":
    main()
