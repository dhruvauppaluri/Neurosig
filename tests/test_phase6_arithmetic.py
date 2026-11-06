"""Phase 6 independent arithmetic verification of fixed-v1.1.

Every expected value here comes from tests/oracles.py (plain ints / Fraction written from the spec), or from
closed-form math (analytic error bounds, constants derived with exact rationals). None of it calls the code under test.
Random seeds are explicit in each test and parameter list.
"""
from fractions import Fraction
from math import floor

import numpy as np
import pytest
from scipy import signal

from scripts.filter_iir import IirConfig, design_sos
from scripts.fixed_point_model import FixedPointConfig, fixed_iir_filter, quantize_sos
from scripts.phase5.fixed_detector import (
    MAD_TO_SIGMA_Q16, NEVER, SIG_MAX, SIG_MIN, V2_SPEC, FEvent, FixedSpec, adc_quantize, calibrate, car_stage,
    detect_channel, filter_stage, merge_events, run_fixed, spatial_veto,
)
from scripts.phase6 import oracles as O

SPEC = V2_SPEC
SPEC_DICT = {"guard_bits": SPEC.guard_bits, "car": "median", "threshold_mult_q8": SPEC.threshold_mult_q8,
             "mad_iterations": SPEC.mad_iterations, "peak_search": SPEC.peak_search, "refractory": SPEC.refractory,
             "veto_window": SPEC.veto_window, "veto_fraction_q8": SPEC.veto_fraction_q8, "merge_window": SPEC.merge_window}
SOS = [tuple(int(v) for v in row) for row in
       quantize_sos(design_sos(IirConfig(sample_rate_hz=30000.0, highpass_hz=200.0, lowpass_hz=4500.0)),
                    FixedPointConfig(signal_bits=24))]


# ------------------------------------------------------------------ constants
def test_q16_constants_are_the_rounded_exact_values():
    mad = Fraction(6744897501960817, 10 ** 16)  # 0.6744897501960817
    assert MAD_TO_SIGMA_Q16 == round(Fraction(65536) / mad)
    assert SPEC.clip_mult_q16 == round(Fraction(6 * 65536) / mad)
    assert (MAD_TO_SIGMA_Q16, SPEC.clip_mult_q16) == (97164, 582983)  # v1 had 97157 / 582964 (arithmetic slip)


def test_internal_lsb_and_adc_lsb_exact():
    assert Fraction(SPEC.adc_lsb_uv_x32, 32) == Fraction(75, 32) == Fraction(2.34375)
    assert SPEC.internal_lsb_uv == 2.34375 / 1024


# ------------------------------------------------------------------------ ADC
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adc_quantize_matches_exact_rational_oracle(seed):
    rng = np.random.default_rng(seed)
    v = np.concatenate([rng.normal(0, 200, 4000), rng.uniform(-2e5, 2e5, 500),
                        (np.arange(-40, 41) + 0.5) * 2.34375,  # exact half-LSB ties
                        [0.0, 1e12, -1e12, 2.34375 * 32767, 2.34375 * -32768, 2.34375 * 32767.5]])
    got = adc_quantize(v[:, None], SPEC)[:, 0]
    want = [O.adc_code(float(x)) for x in v]
    assert got.tolist() == want


def test_adc_ties_round_away_from_zero_and_saturate():
    assert adc_quantize(np.array([[1.171875, -1.171875, 3.515625, -3.515625]]), SPEC).tolist() == [[1, -1, 2, -2]]
    assert adc_quantize(np.array([[1e9, -1e9]]), SPEC).tolist() == [[32767, -32768]]


# --------------------------------------------------------------- IIR / filter
def _filter_int(codes_1ch):
    out, _, stats = filter_stage(np.asarray(codes_1ch, dtype=np.int64)[:, None], SPEC)
    return out[:, 0].tolist(), stats


@pytest.mark.parametrize("seed", range(4))
def test_filter_matches_integer_oracle_random(seed):
    rng = np.random.default_rng(seed)
    kinds = [rng.integers(-512, 512, 3000), np.rint(rng.normal(0, 8, 3000)), np.full(3000, 511), np.full(3000, -512),
             np.where(np.arange(3000) % 2 == 0, 511, -512), np.concatenate([[400], np.zeros(2999)])]
    for x in kinds:
        x = np.asarray(x, dtype=np.int64)
        got, _ = _filter_int(x)
        want = O.biquad_cascade(x.tolist(), SOS)[1]
        assert got == want


def test_filter_input_saturates_to_s24_for_out_of_native_range_codes():
    x = np.array([32767, -32768, 8191, 8192, -8192, -8193] * 50, dtype=np.int64)
    got, _ = _filter_int(x)
    assert got == O.biquad_cascade(x.tolist(), SOS)[1]
    assert O.sat24(32767 * 1024) == SIG_MAX and O.sat24(-32768 * 1024) == SIG_MIN


def test_section_saturation_matches_oracle_and_is_counted():
    gain2 = [((1 << 31) - 1, 0, 0, 1 << 30, 0, 0), (1 << 30, 0, 0, 1 << 30, 0, 0)]  # b0 ~ 2.0: doubles the input
    cfg = FixedPointConfig(signal_bits=24)
    x = (np.array([[6_000_000], [-6_000_000], [1_000_000], [0]] * 5, dtype=np.int64))
    out, stats = fixed_iir_filter(x, np.array(gain2, dtype=np.int64), cfg)
    # oracle with guard_bits=0 (input already in internal units)
    s1, s2 = O.biquad_cascade(x[:, 0].tolist(), gain2, guard_bits=0)
    assert out[:, 0].tolist() == s2
    assert stats[0]["saturated_samples"] > 0
    assert min(s1) == SIG_MIN and max(s1) == SIG_MAX


def _analytic_bound_uv(sos):
    """Worst-case |fixed - exact(quantised coefficients)| from the L1 norm of the truncation error transfer functions."""
    n = 60000
    imp = np.zeros(n); imp[0] = 1.0
    q = np.asarray(sos, dtype=float) / 2 ** 30
    inv_a1 = signal.lfilter([1.0], q[0, 3:6], imp)
    inv_a2 = signal.lfilter([1.0], q[1, 3:6], imp)
    h1 = signal.lfilter(q[1, 0:3], q[1, 3:6], inv_a1)
    return (np.abs(h1).sum() + np.abs(inv_a2).sum()) * 2.34375 / 1024  # LSB-per-truncation -> microvolts


@pytest.mark.parametrize("kind", ["noise", "pinned_positive", "pinned_negative", "nyquist", "step", "burst_then_zero"])
def test_filter_error_vs_float_below_analytic_bound(kind):
    rng = np.random.default_rng(5)
    n = 20000
    x = {"noise": rng.integers(-512, 512, n), "pinned_positive": np.full(n, 400) + rng.integers(0, 3, n),
         "pinned_negative": np.full(n, -400) - rng.integers(0, 3, n),
         "nyquist": np.where(np.arange(n) % 2 == 0, 511, -512), "step": np.where(np.arange(n) > 5000, 450, 0),
         "burst_then_zero": np.concatenate([rng.integers(-512, 512, 2000), np.zeros(n - 2000)])}[kind].astype(np.int64)
    got = np.array(_filter_int(x)[0], dtype=float) * 2.34375 / 1024
    exact = signal.sosfilt(np.asarray(SOS, dtype=float) / 2 ** 30, x * 2.34375)  # float64, quantised coefficients
    bound = _analytic_bound_uv(SOS)
    assert np.abs(got - exact).max() < bound
    assert bound < 1.6  # 1.47 uV for 200/4500 Hz at 10 guard bits; documents the guard-bit requirement


def test_zero_input_stays_zero_and_burst_decays_exactly_to_zero():
    assert set(_filter_int(np.zeros(500, dtype=np.int64))[0]) == {0}
    x = np.concatenate([np.random.default_rng(2).integers(-512, 512, 1500), np.zeros(8000, dtype=np.int64)])
    assert set(_filter_int(x)[0][-500:]) == {0}  # no limit cycle with truncation toward zero


def test_long_run_no_drift_and_bounded_error():
    n = 300_000
    rng = np.random.default_rng(9)
    x = np.rint(rng.normal(0, 10, n) + 60 * np.sin(2 * np.pi * 7 * np.arange(n) / 30000)).astype(np.int64)
    got = np.array(_filter_int(x)[0], dtype=float) * 2.34375 / 1024
    exact = signal.sosfilt(np.asarray(SOS, dtype=float) / 2 ** 30, x * 2.34375)
    err = got - exact
    assert np.abs(err).max() < _analytic_bound_uv(SOS)
    assert abs(err[: n // 3].mean() - err[-n // 3:].mean()) < 0.2  # no secular drift of the error


def test_state_is_reset_between_runs():
    x = np.random.default_rng(3).integers(-512, 512, 2000)
    assert _filter_int(x)[0] == _filter_int(x)[0]
    y = np.concatenate([x, x])
    assert _filter_int(y)[0][2000:] != _filter_int(x)[0]  # second half starts from non-zero state; fresh run starts from zero


# ------------------------------------------------------------------------ CAR
@pytest.mark.parametrize("n_ch", [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 32, 33])
@pytest.mark.parametrize("mode", ["median", "mean"])
def test_car_matches_oracle_for_channel_counts_and_masks(n_ch, mode):
    rng = np.random.default_rng(100 + n_ch)
    f = rng.integers(-2 ** 23, 2 ** 23, (40, n_ch)).astype(np.int64)
    f[0] = SIG_MAX; f[1] = SIG_MIN; f[2, ::2] = 5; f[2, 1::2] = -5  # extremes and tie patterns
    active = rng.random(n_ch) > 0.3
    active[0] = True
    got = car_stage(f, active, mode)
    for t in range(f.shape[0]):
        ref = O.car_ref(f[t].tolist(), active.tolist(), mode)
        assert got[t].tolist() == [O.sat24(int(v) - ref) for v in f[t]]


def test_mean_car_within_one_lsb_of_exact_rounding_and_exact_for_powers_of_two():
    rng = np.random.default_rng(7)
    for n in (2, 3, 5, 6, 7, 10, 31, 32, 33):
        worst = 0
        for _ in range(300):
            fr = rng.integers(-2 ** 23, 2 ** 23, n).tolist()
            a = O.car_ref(fr, [True] * n, "mean"); b = O.car_mean_exact_round_half_up(fr, [True] * n)
            worst = max(worst, abs(a - b))
        assert worst <= 1 and (worst == 0 if n in (2, 32) else True)


def test_median_even_count_floors_negative_midpoint():
    assert O.car_ref([-3, -2, -1, 0], [True] * 4, "median") == -2
    f = np.array([[-3, -2, -1, 0]], dtype=np.int64)
    assert car_stage(f, np.ones(4, bool), "median")[0].tolist() == [-1, 0, 1, 2]


# ---------------------------------------------------------------- calibration
SERIES = {
    "gauss": lambda r: np.rint(r.normal(0, 4000, 5000)),
    "heavy_tail": lambda r: np.rint(r.standard_cauchy(5000) * 800).clip(-2 ** 23, 2 ** 23 - 1),
    "outliers": lambda r: np.concatenate([r.normal(0, 2000, 4900), r.normal(0, 400000, 100)]).round(),
    "bimodal": lambda r: np.concatenate([r.normal(-9000, 500, 2500), r.normal(9000, 500, 2500)]).round(),
    "offset": lambda r: np.rint(r.normal(120000, 3000, 5000)),
    "tiny": lambda r: r.integers(-1, 2, 5000).astype(float),
    "odd_len": lambda r: np.rint(r.normal(0, 3000, 4999)),
    "short": lambda r: np.rint(r.normal(0, 3000, 40)),
    "extreme": lambda r: r.choice([SIG_MIN, SIG_MAX, 0], 3000).astype(float),
}


@pytest.mark.parametrize("name", list(SERIES))
@pytest.mark.parametrize("seed", [0, 1])
def test_calibration_matches_oracle(name, seed):
    x = SERIES[name](np.random.default_rng(seed)).astype(np.int64)
    got = calibrate(x[:, None], np.array([True]), SPEC)
    want = O.calibrate(x.tolist())
    assert (int(got["baseline"][0]), int(got["mad"][0]), int(got["sigma_q16"][0]), int(got["threshold"][0]),
            int(got["recip_q"][0]), bool(got["armed"][0])) == (want["baseline"], want["mad"], want["sigma_q16"],
                                                              want["thr"], want["recip"], want["armed"])


def test_calibration_zero_mad_disarms_channel_and_never_divides():
    for x in (np.zeros(500, dtype=np.int64), np.full(500, 77, dtype=np.int64)):
        c = calibrate(x[:, None], np.array([True]), SPEC)
        assert not c["armed"][0] and c["threshold"][0] == NEVER and c["recip_q"][0] == 0
    assert O.calibrate([0] * 100)["thr"] == NEVER


def test_threshold_offset_rounds_half_up_exact_tie():
    x = np.array([2048, -2048] * 400, dtype=np.int64)  # baseline 0, MAD exactly 2048 -> offset tie at x.5
    want = O.calibrate(x.tolist())
    assert want["mad"] == 2048 and Fraction(1024 * want["sigma_q16"], 1 << 24).denominator == 2  # exactly k + 1/2
    got = calibrate(x[:, None], np.array([True]), SPEC)
    assert int(got["threshold"][0]) == want["thr"]


def test_disabled_channel_is_unarmed():
    x = np.random.default_rng(1).integers(-3000, 3000, (500, 2)).astype(np.int64)
    c = calibrate(x, np.array([True, False]), SPEC)
    assert c["armed"].tolist() == [True, False] and c["threshold"][1] == NEVER


# ------------------------------------------------------- detection / merge / veto
@pytest.mark.parametrize("seed", range(12))
def test_detect_channel_matches_oracle_with_ties_boundaries_and_gaps(seed):
    rng = np.random.default_rng(1000 + seed)
    n = 1500
    trace = rng.integers(-6, 7, n).astype(np.int64)  # tiny alphabet: many equalities / tied minima
    thr = int(rng.integers(-5, 0)); base = int(rng.integers(0, 3)); recip = int(rng.integers(1, 2 ** 31))
    gaps = rng.random(n) > 0.05 if seed % 2 else np.ones(n, dtype=bool)
    ts = np.arange(n)[gaps]
    tr = trace[gaps]
    got = detect_channel(tr, ts, thr, base, recip, 3, SPEC)
    want = O.detect(tr.tolist(), ts.tolist(), thr, base, recip, SPEC.peak_search, SPEC.refractory)
    assert [(e.sample, e.code, e.delta, e.amp_q8) for e in got] == want


@pytest.mark.parametrize("seed", range(10))
def test_veto_and_merge_match_oracle(seed):
    rng = np.random.default_rng(2000 + seed)
    n_ch = int(rng.integers(2, 33))
    n = int(rng.integers(0, 120))
    ev = [FEvent(int(rng.integers(0, 400)), int(rng.integers(0, n_ch)), -int(rng.integers(1, 99)), int(rng.integers(1, 99)),
                 int(rng.integers(0, 6))) for _ in range(n)]
    ev = list({(e.sample, e.channel): e for e in ev}.values())  # one candidate per (sample, channel)
    tup = [(e.sample, e.channel, e.code, e.delta, e.amp_q8) for e in ev]
    kept, _ = spatial_veto(ev, n_ch, SPEC)
    assert sorted((e.sample, e.channel, e.code, e.delta, e.amp_q8) for e in kept) == sorted(O.veto(tup, n_ch))
    got = [(e.sample, e.channel, e.code, e.delta, e.amp_q8) for e in merge_events(kept, SPEC)]
    assert got == O.merge(O.veto(tup, n_ch))


def test_veto_group_size_threshold_by_channel_count():
    for n_active in (2, 3, 4, 7, 8, 31, 32):
        need = max(2, -(-(128 * n_active) // 256))
        ev = [FEvent(100, c, -9, 9, 500) for c in range(need)]
        kept, rej = spatial_veto(ev, n_active, SPEC)
        assert kept == [] and rej == need
        kept, rej = spatial_veto(ev[:need - 1], n_active, SPEC)
        assert len(kept) == need - 1 and rej == 0


# ---------------------------------------------------- full pipeline vs oracle
@pytest.mark.parametrize("seed,n_ch", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (6, 8), (7, 4)])
def test_full_pipeline_matches_independent_oracle(seed, n_ch):
    rng = np.random.default_rng(3000 + seed)
    n = 2200
    x = np.rint(rng.normal(0, 4, (n, n_ch))).astype(np.int64)
    for s in range(300, n - 100, 260):
        c = int(rng.integers(0, n_ch)); t = np.arange(-6, 12)
        x[s + t, c] += np.rint(-70 * np.exp(-0.5 * (t / 2.0) ** 2)).astype(np.int64)
    x = np.clip(x, -512, 511)
    active = rng.random(n_ch) > 0.25; active[0] = True
    valid = rng.random(n) > 0.02; valid[:3] = True
    mode = "mean" if seed % 3 == 0 else "median"
    spec = FixedSpec(highpass_hz=200.0, lowpass_hz=4500.0, car=mode)
    r = run_fixed(x, spec, active=active, valid=valid)
    d = dict(SPEC_DICT, car=mode)
    want_events, want_cands = O.pipeline(x.tolist(), SOS, active.tolist(), valid.tolist(), d)
    assert [(e.sample, e.channel, e.code, e.delta, e.amp_q8) for e in r["candidates"]] == want_cands
    assert [(e.sample, e.channel, e.code, e.delta, e.amp_q8) for e in r["events"]] == want_events


def test_full_pipeline_32_channels_short_matches_oracle():
    rng = np.random.default_rng(77)
    n = 700
    x = np.clip(np.rint(rng.normal(0, 5, (n, 32)) + 0.0), -512, 511).astype(np.int64)
    for s in (200, 450):
        x[s:s + 12, 9] += np.rint(-90 * np.exp(-0.5 * ((np.arange(12) - 4) / 2.0) ** 2)).astype(np.int64)
    r = run_fixed(x, SPEC)
    ev, ca = O.pipeline(x.tolist(), SOS, [True] * 32, [True] * n, SPEC_DICT)
    assert [(e.sample, e.channel, e.code, e.delta, e.amp_q8) for e in r["candidates"]] == ca
    assert [(e.sample, e.channel, e.code, e.delta, e.amp_q8) for e in r["events"]] == ev


def test_amp_q8_and_saturation_register():
    assert O.amp_q8(1 << 23, (1 << 32) - 1) == 0xFFFF
    assert O.amp_q8(5, 0) == 0
