"""Stress and corner-case tests for the Phase 5 integer reference (fixed-v1).

The known-good regression group (``test_regression_*``) must pass before a final
implementation is selected. Randomized tests use fixed seeds.
"""

import numpy as np
import pytest

from scripts.phase5.fixed_detector import (
    NEVER, SIG_MAX, SIG_MIN, FEvent, FixedSpec, StreamingFixed, adc_quantize, amp_q8, calibrate,
    car_stage, detect_channel, median_int_1d, merge_events, run_fixed, spatial_veto,
)

SPEC = FixedSpec()


def make_codes(seed, n=5000, ch=8, noise=4.0, cm=0.0, spikes=6, amp=-60):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, noise, (n, ch)) + rng.normal(0, cm, (n, 1)) if cm else rng.normal(0, noise, (n, ch))
    for s in np.linspace(500, n - 300, spikes).astype(int):
        c = int(rng.integers(0, ch))
        t = np.arange(-6, 12)
        x[s + t, c] += amp * np.exp(-0.5 * (t / 2.0) ** 2)
    return np.clip(np.rint(x), -512, 511).astype(np.int64)


def stream_events(codes, r, spec=SPEC, active=None, valid=None):
    n_ch = codes.shape[1]
    s = StreamingFixed(spec, r["sos"], r["calibration"], np.ones(n_ch, bool) if active is None else active, n_ch)
    ev = []
    for t in range(codes.shape[0]):
        if valid is not None and not valid[t]:
            ev += s.push(None, timestamp=t)
        else:
            ev += s.push(codes[t], timestamp=t)
    ev += s.flush()
    return sorted(ev, key=lambda e: (e.sample, e.channel))


# ------------------------------------------------------------ regression set
@pytest.mark.parametrize("seed,ch", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 8), (5, 32)])
def test_regression_streaming_matches_vectorised(seed, ch):
    codes = make_codes(seed, n=4000 if ch == 32 else 6000, ch=ch)
    r = run_fixed(codes)
    assert stream_events(codes, r) == r["candidates"]


def test_regression_deterministic_and_reset_state():
    codes = make_codes(11)
    a, b = run_fixed(codes), run_fixed(codes)
    assert a["events"] == b["events"] and a["candidates"] == b["candidates"]
    assert np.array_equal(a["car"], b["car"])
    s1 = stream_events(codes, a)
    s2 = stream_events(codes, a)  # fresh instance each call = reset state
    assert s1 == s2


def test_regression_median_int_even_odd_negative_floor():
    assert median_int_1d(np.array([5, 1, 3])) == 3
    assert median_int_1d(np.array([1, 2, 3, 4])) == 2  # floor((2+3)/2)
    assert median_int_1d(np.array([-3, -2, -1, 0])) == -2  # floor(-1.5) = -2
    rng = np.random.default_rng(0)
    for n in (5, 6, 31, 32):
        x = rng.integers(-1000, 1000, n)
        s = np.sort(x)
        want = int(s[n // 2]) if n % 2 else (int(s[n // 2 - 1]) + int(s[n // 2])) // 2
        assert median_int_1d(x) == want


def test_regression_car_median_and_mean_exact_small_cases():
    f = np.array([[1, 2, 3, 100], [-4, -3, 7, 9]], dtype=np.int64)
    out = car_stage(f, np.ones(4, bool), "median")
    assert out[0].tolist() == [1 - 2, 2 - 2, 3 - 2, 100 - 2]  # floor((2+3)/2)=2
    assert out[1].tolist() == [-4 - 2, -3 - 2, 7 - 2, 9 - 2]  # floor((-3+7)/2)=2
    out = car_stage(np.array([[8, 8, 8, 9]], dtype=np.int64), np.ones(4, bool), "mean")
    assert out[0].tolist() == [8 - 8, 8 - 8, 8 - 8, 9 - 8]  # round(33/4)=8 (half-up of 8.25)


def test_regression_threshold_equality_boundary():
    trace = np.array([10, 10, 5, 3, 4, 10, 10, 10, 10, 10] + [10] * 40, dtype=np.int64)
    ts = np.arange(trace.size)
    ev = detect_channel(trace, ts, 5, 10, 1 << 20, 0, SPEC)  # value == 5 is not below threshold
    assert [e.sample for e in ev] == [3]  # crossing at index 3 (value 3 < 5), trough 3
    at = trace.copy(); at[3] = 5; at[4] = 5
    assert detect_channel(at, ts, 5, 10, 1 << 20, 0, SPEC) == []
    just = trace.copy(); just[2] = 4
    assert detect_channel(just, ts, 5, 10, 1 << 20, 0, SPEC)[0].sample == 3  # first crossing 4<5 at idx2 -> trough 3


def test_regression_refractory_boundary_exact():
    R = SPEC.refractory
    def trace_with(gap):
        t = np.full(400, 10, dtype=np.int64)
        t[100] = 0
        t[100 + gap] = 0
        return t
    ts = np.arange(400)
    assert len(detect_channel(trace_with(R), ts, 5, 10, 1 << 20, 0, SPEC)) == 2  # exactly R apart allowed
    assert len(detect_channel(trace_with(R - 1), ts, 5, 10, 1 << 20, 0, SPEC)) == 1


def test_regression_recording_start_and_end():
    t = np.full(60, 10, dtype=np.int64)
    t[0], t[1] = 10, 0  # crossing at index 1 (first usable)
    assert [e.sample for e in detect_channel(t, np.arange(60), 5, 10, 1 << 20, 0, SPEC)] == [1]
    end = np.full(60, 10, dtype=np.int64)
    end[57] = 0
    end[59] = -3  # window truncated by end of stream; deeper trough at the very last sample
    assert [e.sample for e in detect_channel(end, np.arange(60), 5, 10, 1 << 20, 0, SPEC)] == [59]


# --------------------------------------------------------------- corner cases
def test_zero_input_has_no_events_and_unarmed_channels():
    codes = np.zeros((3000, 4), dtype=np.int64)
    r = run_fixed(codes)
    assert r["events"] == [] and not r["calibration"]["armed"].any()
    assert (r["calibration"]["threshold"] == NEVER).all()


def test_constant_offset_decays_no_events():
    codes = np.full((6000, 4), 300, dtype=np.int64)
    r = run_fixed(codes)
    assert r["events"] == []
    assert int(np.abs(r["filtered"][-500:]).max()) <= 2  # zero-input style decay of the HP section


def test_max_min_adc_codes_no_overflow_and_in_range():
    for v in (-32768, 32767):
        codes = np.full((2000, 3), v, dtype=np.int64)
        r = run_fixed(codes)
        assert r["filtered"].min() >= SIG_MIN and r["filtered"].max() <= SIG_MAX
        assert r["car"].min() >= SIG_MIN and r["car"].max() <= SIG_MAX
    alt = np.where(np.arange(3000)[:, None] % 2 == 0, 32767, -32768) * np.ones((1, 3), dtype=np.int64)
    r = run_fixed(alt)  # worst-case alternating input must not raise
    assert r["filtered"].min() >= SIG_MIN


def test_adc_saturates_at_int16():
    c = adc_quantize(np.array([[1e9, -1e9, 0.0]]), SPEC)
    assert c.tolist() == [[32767, -32768, 0]]


def test_large_common_mode_is_removed_by_median_car():
    rng = np.random.default_rng(3)
    n, ch = 6000, 8
    cm = (200 * np.sin(2 * np.pi * 800 * np.arange(n) / 30000))[:, None] * np.ones((1, ch))
    codes = np.clip(np.rint(rng.normal(0, 4, (n, ch)) + cm), -512, 511).astype(np.int64)
    with_car = run_fixed(codes, FixedSpec(car="median"))
    without = run_fixed(codes, FixedSpec(car="none"))
    # the shared 800 Hz interference dominates every channel without CAR and is removed with it
    assert with_car["car"].std() < 0.1 * without["car"].std()
    assert np.abs(with_car["car"].mean(axis=1)).max() < 0.2 * np.abs(without["car"]).max()


def test_noisy_channel_gets_its_own_threshold():
    codes = make_codes(5, ch=4)
    codes = codes.copy()
    rng = np.random.default_rng(1)
    codes[:, 2] = np.clip(np.rint(codes[:, 2] + rng.normal(0, 30, codes.shape[0])), -512, 511)
    cal = run_fixed(codes)["calibration"]
    assert cal["mad"][2] > 3 * cal["mad"][[0, 1, 3]].max()
    assert cal["threshold"][2] < cal["threshold"][0] < 0 or cal["threshold"][2] <= cal["threshold"][0]


def test_disabled_channels_never_fire_and_are_excluded_from_reference():
    codes = make_codes(7, ch=6, spikes=10)
    active = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
    r = run_fixed(codes, active=active)
    assert all(e.channel in (0, 1, 3, 5) for e in r["candidates"])
    assert stream_events(codes, r, active=active) == r["candidates"]
    # median over active channels only: disabling a wild channel changes the reference
    wild = codes.copy(); wild[:, 2] = 400
    a = run_fixed(wild, active=active)["car"][:, [0, 1, 3, 5]]
    b = run_fixed(wild)["car"][:, [0, 1, 3, 5]]
    assert not np.array_equal(a, b)


def test_dropped_samples_streaming_matches_vectorised_and_timestamps_kept():
    codes = make_codes(9, n=6000, ch=4)
    rng = np.random.default_rng(2)
    valid = rng.random(codes.shape[0]) > 0.01
    valid[:3] = True
    r = run_fixed(codes, valid=valid)
    assert stream_events(codes, r, valid=valid) == r["candidates"]
    assert all(valid[e.sample] for e in r["candidates"])


def test_simultaneous_distant_events_merge_to_strongest():
    ev = [FEvent(100, 2, -50, 50, 600), FEvent(104, 9, -90, 90, 900), FEvent(300, 1, -40, 40, 500)]
    out = merge_events(ev, SPEC)
    assert [(e.sample, e.channel) for e in out] == [(104, 9), (300, 1)]


def test_merge_ties_keep_earliest_then_lowest_channel():
    ev = [FEvent(100, 5, 0, 1, 700), FEvent(100, 3, 0, 1, 700), FEvent(103, 1, 0, 1, 700)]
    out = merge_events(ev, SPEC)
    assert len(out) == 1 and (out[0].sample, out[0].channel) == (100, 3)


def test_artifact_veto_threshold_by_active_channel_count():
    group = [FEvent(100, c, -9, 9, 500) for c in range(16)]
    kept, rej = spatial_veto(group, 32, SPEC)  # need = ceil(0.5*32) = 16 -> rejected
    assert kept == [] and rej == 16
    kept, rej = spatial_veto(group[:15], 32, SPEC)
    assert len(kept) == 15 and rej == 0
    kept, rej = spatial_veto(group[:2], 3, SPEC)  # need = max(2, ceil(1.5)) = 2
    assert kept == [] and rej == 2
    kept, rej = spatial_veto(group[:1], 1, SPEC)  # single channel: need >= 2, never vetoed
    assert len(kept) == 1


def test_overlapping_spikes_same_channel_collapse_by_refractory():
    codes = np.zeros((4000, 2), dtype=np.int64)
    rng = np.random.default_rng(4)
    codes = np.rint(rng.normal(0, 3, codes.shape)).astype(np.int64)
    for s in (1000, 1012):  # 0.4 ms apart on channel 0
        t = np.arange(-6, 10)
        codes[s + t, 0] += np.rint(-80 * np.exp(-0.5 * (t / 2.0) ** 2)).astype(np.int64)
    r = run_fixed(codes)
    near = [e for e in r["candidates"] if e.channel == 0 and 980 < e.sample < 1060]
    assert len(near) == 1


def test_amplitude_register_saturates_16_bits():
    assert amp_q8(1 << 23, (1 << 32) - 1) == 0xFFFF


def test_calibration_matches_gaussian_noise_sigma():
    rng = np.random.default_rng(8)
    x = np.rint(rng.normal(0, 5000, (40000, 3))).astype(np.int64)  # internal-LSB units
    cal = calibrate(x, np.ones(3, bool), SPEC)
    sigma = cal["sigma_q16"] / 65536.0
    assert np.allclose(sigma, 5000, rtol=0.03)
    assert (cal["threshold"] < 0).all()


@pytest.mark.parametrize("seed", range(5))
def test_randomized_streaming_equivalence_with_gaps_and_disabled(seed):
    rng = np.random.default_rng(100 + seed)
    ch = int(rng.integers(2, 9))
    codes = make_codes(200 + seed, n=3000, ch=ch, spikes=8)
    active = rng.random(ch) > 0.25
    active[0] = True
    valid = rng.random(codes.shape[0]) > 0.02
    valid[:2] = True
    r = run_fixed(codes, active=active, valid=valid)
    assert stream_events(codes, r, active=active, valid=valid) == r["candidates"]
