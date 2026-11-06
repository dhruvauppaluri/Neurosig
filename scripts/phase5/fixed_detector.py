"""Bit-accurate integer reference for the Phase 5 real-signal detector ("fixed-v1").

Everything between the ADC codes and the emitted events is integer arithmetic
with explicit widths, rounding and saturation. The IIR section reuses
``scripts.fixed_point_model`` unchanged (Q2.30 coefficients, 24-bit signed
states, 64-bit accumulators, truncation toward zero, saturating section output).

Two independent implementations live here and must agree bit for bit:
  * ``run_fixed``           vectorised numpy reference (used for benchmarking)
  * ``StreamingFixed``      scalar pure-Python-int, frame-by-frame state machine
                             (what an RTL pipeline would actually do)

Specification: docs/phase5-fixed-point-spec.md. No RTL exists for this model;
nothing here claims RTL verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from scripts.filter_iir import IirConfig, design_sos
from scripts.fixed_point_model import FixedPointConfig, fixed_iir_filter, quantize_sos
from scripts.quantize_adc import round_half_away_from_zero

SIGNAL_BITS = 24
SIG_MIN = -(1 << (SIGNAL_BITS - 1))
SIG_MAX = (1 << (SIGNAL_BITS - 1)) - 1
MAD_TO_SIGMA_Q16 = 97157  # round(65536 / 0.6744897501960817)
NEVER = SIG_MIN  # threshold sentinel: x < SIG_MIN is impossible, channel can never fire


@dataclass(frozen=True)
class FixedSpec:
    version: str = "fixed-v1"
    sample_rate_hz: int = 30_000
    adc_lsb_uv_x32: int = 75  # 75/32 = 2.34375 uV per code (native 10-bit probe ADC)
    adc_bits: int = 16  # int16 container, saturating
    guard_bits: int = 10  # internal LSB = adc LSB / 1024 = 0.002288818359375 uV (G=4 gave 3 uV RMS filter error)
    highpass_hz: float = 300.0
    lowpass_hz: float = 6000.0
    car: str = "median"  # median | mean | none
    threshold_mult_q8: int = 1024  # 4.0 sigma in Q8
    clip_mult_q16: int = 582_964  # round(65536 * 6 / 0.6744897501960817) = 8.8956 MADs
    mad_iterations: int = 3
    peak_search: int = 15  # samples (0.5 ms)
    refractory: int = 30  # samples (1.0 ms)
    merge_window: int = 15  # samples (0.5 ms)
    veto_window: int = 6  # samples (0.2 ms)
    veto_fraction_q8: int = 128  # 0.5; 0 disables the veto

    @property
    def internal_lsb_uv(self) -> float:
        return self.adc_lsb_uv_x32 / 32.0 / (1 << self.guard_bits)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FEvent:
    sample: int  # timestamp counter value of the trough
    channel: int
    code: int  # CAR output at the trough (signed, internal LSB)
    delta: int  # baseline - code, > 0, internal LSB
    amp_q8: int  # delta / sigma in Q8.8 sigma units, saturated to 16 bits


# ------------------------------------------------------------------- helpers
def round_shift_right(v, bits: int):
    """Round half up arithmetic shift: (v + 2**(bits-1)) >> bits."""
    return (v + (1 << (bits - 1))) >> bits


def floor_avg(a, b):
    return (a + b) >> 1


def sat24(v):
    return np.clip(v, SIG_MIN, SIG_MAX)


def median_int_1d(x: np.ndarray) -> int:
    """Exact integer median; even counts use floor of the mean of the two middle values."""
    s = np.sort(np.asarray(x, dtype=np.int64))
    n = s.size
    if n % 2:
        return int(s[n // 2])
    return int((int(s[n // 2 - 1]) + int(s[n // 2])) >> 1)


def adc_quantize(voltages_uv: np.ndarray, spec: FixedSpec) -> np.ndarray:
    """volts -> int16 codes, round half away from zero, saturate."""
    lsb = spec.adc_lsb_uv_x32 / 32.0
    codes = round_half_away_from_zero(np.asarray(voltages_uv, dtype=np.float64) / lsb)
    lim = (1 << (spec.adc_bits - 1))
    return np.clip(codes, -lim, lim - 1).astype(np.int64)


def filter_stage(adc_codes: np.ndarray, spec: FixedSpec):
    fcfg = FixedPointConfig(signal_bits=SIGNAL_BITS)
    sos = quantize_sos(design_sos(IirConfig(sample_rate_hz=float(spec.sample_rate_hz),
                                            highpass_hz=spec.highpass_hz, lowpass_hz=spec.lowpass_hz)), fcfg)
    x = np.asarray(adc_codes, dtype=np.int64) << spec.guard_bits
    out, stats = fixed_iir_filter(x, sos, fcfg)
    return out.astype(np.int64), sos, stats


def car_stage(filtered: np.ndarray, active: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return filtered.copy()
    cols = np.flatnonzero(active)
    n = cols.size
    if n == 0:
        raise ValueError("no active channels")
    sub = filtered[:, cols]
    if method == "median":
        s = np.sort(sub, axis=1)
        ref = s[:, n // 2] if n % 2 else (s[:, n // 2 - 1] + s[:, n // 2]) >> 1
    elif method == "mean":
        recip = (1 << 32) // n if (n & (n - 1)) == 0 else int(round((1 << 32) / n))
        ref = (sub.sum(axis=1) * recip + (1 << 31)) >> 32
    else:
        raise ValueError(method)
    return sat24(filtered - ref[:, None])


def calibrate(car_out: np.ndarray, active: np.ndarray, spec: FixedSpec) -> dict:
    """Offline integer calibration of baseline / MAD / threshold registers.

    Matches the float reference: median baseline, iterative MAD with 6-sigma clipping.
    """
    n_ch = car_out.shape[1]
    baseline = np.zeros(n_ch, dtype=np.int64)
    mad = np.zeros(n_ch, dtype=np.int64)
    for c in range(n_ch):
        x = car_out[:, c].astype(np.int64)
        b = median_int_1d(x)
        m = median_int_1d(np.abs(x - b))
        for _ in range(spec.mad_iterations):
            lim = (m * spec.clip_mult_q16) >> 16
            keep = np.abs(x - b) <= lim
            if int(keep.sum()) < 32:
                keep = np.ones(x.size, dtype=bool)
            xs = x[keep]
            b = median_int_1d(xs)
            m = median_int_1d(np.abs(xs - b))
        baseline[c], mad[c] = b, m
    sigma_q16 = mad * MAD_TO_SIGMA_Q16
    off = round_shift_right(spec.threshold_mult_q8 * sigma_q16, 24)
    thr = baseline - off
    ok = active & (mad > 0)
    thr = np.where(ok, np.clip(thr, SIG_MIN + 1, SIG_MAX), NEVER)
    recip = np.where(ok, (1 << 48) // np.maximum(sigma_q16, 1), 0)
    return {"baseline": baseline, "mad": mad, "sigma_q16": sigma_q16, "threshold": thr.astype(np.int64),
            "recip_q": recip.astype(np.int64), "armed": ok}


# ----------------------------------------------------------------- detection
def amp_q8(delta: int, recip: int) -> int:
    return min(0xFFFF, (delta * recip) >> 24)


def detect_channel(trace: np.ndarray, ts: np.ndarray, thr: int, baseline: int, recip: int,
                   channel: int, spec: FixedSpec) -> list[FEvent]:
    """Per-channel crossing / trough / refractory on a timestamped sample list."""
    events: list[FEvent] = []
    if thr == NEVER:
        return events
    below = trace < thr
    cross = np.flatnonzero(~below[:-1] & below[1:]) + 1
    last_peak = -spec.refractory - 1
    for j in cross:
        t_cross = int(ts[j])
        if t_cross - last_peak < spec.refractory:
            continue
        stop = int(np.searchsorted(ts, t_cross + spec.peak_search, side="right"))
        seg = trace[j:stop]
        k = j + int(np.argmin(seg))  # first minimum on ties
        t_peak = int(ts[k])
        if t_peak - last_peak < spec.refractory:
            continue
        code = int(trace[k])
        delta = baseline - code
        events.append(FEvent(t_peak, channel, code, delta, amp_q8(delta, recip)))
        last_peak = t_peak
    return events


def spatial_veto(events: Sequence[FEvent], n_active: int, spec: FixedSpec) -> tuple[list[FEvent], int]:
    if not events or spec.veto_fraction_q8 <= 0:
        return list(events), 0
    need = max(2, (spec.veto_fraction_q8 * n_active + 255) >> 8)  # ceil(frac * N)
    ordered = sorted(events, key=lambda e: (e.sample, e.channel))
    kept: list[FEvent] = []
    rejected = 0
    i = 0
    while i < len(ordered):
        start = ordered[i].sample
        group = [ordered[i]]
        i += 1
        while i < len(ordered) and ordered[i].sample - start <= spec.veto_window:
            group.append(ordered[i])
            i += 1
        if len({e.channel for e in group}) >= need:
            rejected += len(group)
            continue
        kept.extend(group)
    return kept, rejected


def merge_events(events: Sequence[FEvent], spec: FixedSpec) -> list[FEvent]:
    """Groups anchored on the first event; keep the largest amp_q8 (first on ties)."""
    if not events:
        return []
    groups: list[list[FEvent]] = []
    for e in sorted(events, key=lambda c: (c.sample, c.channel)):
        if not groups or e.sample - groups[-1][0].sample > spec.merge_window:
            groups.append([e])
        else:
            groups[-1].append(e)
    out = []
    for g in groups:
        best = g[0]
        for e in g[1:]:
            if e.amp_q8 > best.amp_q8:
                best = e
        out.append(best)
    return out


# ------------------------------------------------------------------ pipeline
def run_fixed(adc_codes: np.ndarray, spec: FixedSpec | None = None, active: np.ndarray | None = None,
              valid: np.ndarray | None = None, calibration: dict | None = None) -> dict:
    """Vectorised reference. ``valid`` marks present frames (False = dropped sample).

    Dropped frames: filter state holds, no comparison, timestamp counter still
    advances. Without ``valid`` timestamps are 0..N-1.
    """
    spec = spec or FixedSpec()
    codes = np.asarray(adc_codes, dtype=np.int64)
    n_t, n_ch = codes.shape
    active = np.ones(n_ch, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    ts_all = np.arange(n_t, dtype=np.int64)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        codes = codes[valid]
        ts = ts_all[valid]
    else:
        ts = ts_all
    filtered, sos, fstats = filter_stage(codes, spec)
    car = car_stage(filtered, active, spec.car)
    cal = calibration or calibrate(car, active, spec)
    cands: list[FEvent] = []
    for c in range(n_ch):
        if not active[c]:
            continue
        cands.extend(detect_channel(car[:, c], ts, int(cal["threshold"][c]), int(cal["baseline"][c]),
                                    int(cal["recip_q"][c]), c, spec))
    cands.sort(key=lambda e: (e.sample, e.channel))
    n_active = int(active.sum())
    cleaned, vetoed = spatial_veto(cands, n_active, spec)
    merged = merge_events(cleaned, spec)
    return {"events": merged, "candidates": cands, "n_vetoed": vetoed, "filtered": filtered, "car": car,
            "calibration": cal, "sos": sos, "filter_stats": fstats, "timestamps": ts}


# --------------------------------------------------- scalar streaming model
class StreamingFixed:
    """Frame-by-frame pure-Python-int model of filter -> CAR -> per-channel detector.

    Calibration registers are an input (offline). ``push`` takes one frame of ADC
    codes (or None for a dropped frame) and returns candidate events whose peak
    search window just closed. ``flush`` closes any open window (end of stream /
    gap). Merge and veto operate on the emitted event list exactly as in run_fixed.
    """

    def __init__(self, spec: FixedSpec, sos_q30: np.ndarray, calibration: dict, active: np.ndarray, n_channels: int):
        self.spec = spec
        self.sos = [tuple(int(v) for v in row) for row in np.asarray(sos_q30)]
        self.n = n_channels
        self.active = [bool(a) for a in active]
        self.cols = [c for c in range(n_channels) if self.active[c]]
        self.thr = [int(v) for v in calibration["threshold"]]
        self.base = [int(v) for v in calibration["baseline"]]
        self.recip = [int(v) for v in calibration["recip_q"]]
        # per channel per section: x1, x2, y1, y2
        self.st = [[[0, 0, 0, 0] for _ in self.sos] for _ in range(n_channels)]
        self.t = -1
        self.prev = [None] * n_channels  # previous valid CAR output
        self.last_peak = [-spec.refractory - 1] * n_channels
        self.open = [None] * n_channels  # (t_cross, best_t, best_code)

    @staticmethod
    def _trunc_shift(v: int, bits: int) -> int:
        return v >> bits if v >= 0 else -((-v) >> bits)

    def _filter_channel(self, c: int, x: int) -> int:
        for si, (b0, b1, b2, _a0, a1, a2) in enumerate(self.sos):
            x1, x2, y1, y2 = self.st[c][si]
            acc = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y = self._trunc_shift(acc, 30)
            y = SIG_MIN if y < SIG_MIN else SIG_MAX if y > SIG_MAX else y
            self.st[c][si] = [x, x1, y, y1]
            x = y
        return x

    def _car(self, ys: list[int]) -> int:
        act = sorted(ys[c] for c in self.cols)
        n = len(act)
        if self.spec.car == "none":
            return 0
        if self.spec.car == "median":
            return act[n // 2] if n % 2 else (act[n // 2 - 1] + act[n // 2]) >> 1
        recip = (1 << 32) // n if (n & (n - 1)) == 0 else int(round((1 << 32) / n))
        return (sum(act) * recip + (1 << 31)) >> 32

    def _close(self, c: int, out: list[FEvent]) -> None:
        o = self.open[c]
        self.open[c] = None
        if o is None:
            return
        _, t_peak, code = o
        if t_peak - self.last_peak[c] < self.spec.refractory:
            return
        delta = self.base[c] - code
        out.append(FEvent(t_peak, c, code, delta, amp_q8(delta, self.recip[c])))
        self.last_peak[c] = t_peak

    def push(self, frame, timestamp: int | None = None) -> list[FEvent]:
        self.t = self.t + 1 if timestamp is None else timestamp
        out: list[FEvent] = []
        for c in range(self.n):  # close windows that expire (also on dropped frames)
            o = self.open[c]
            if o is not None and self.t > o[0] + self.spec.peak_search:
                self._close(c, out)
        if frame is None:
            return out
        ys = [self._filter_channel(c, int(frame[c]) << self.spec.guard_bits) for c in range(self.n)]
        ref = self._car(ys)
        for c in self.cols:
            v = ys[c] - ref
            v = SIG_MIN if v < SIG_MIN else SIG_MAX if v > SIG_MAX else v
            o = self.open[c]
            if o is not None:
                if v < o[2]:
                    self.open[c] = (o[0], self.t, v)
            elif (self.thr[c] != NEVER and self.prev[c] is not None and self.prev[c] >= self.thr[c] and v < self.thr[c]
                  and self.t - self.last_peak[c] >= self.spec.refractory):
                self.open[c] = (self.t, self.t, v)
            self.prev[c] = v
        return out

    def flush(self) -> list[FEvent]:
        out: list[FEvent] = []
        for c in range(self.n):
            self._close(c, out)
        return out
