"""Independent oracles for the fixed-v1.1 arithmetic.

Written from docs/phase6-rtl-contract.md using plain Python integers and fractions.Fraction. They share NO code
with scripts/phase5/fixed_detector.py (no numpy, no helper reuse): shifts are expressed as floor division, medians use
sorted lists, rounding uses exact rational arithmetic. A match therefore means two separate implementations of the spec
agree, not that one function was checked against itself.
"""
from fractions import Fraction
from math import floor

S24_MIN, S24_MAX = -(1 << 23), (1 << 23) - 1
NEVER = S24_MIN


def adc_code(v_uv: float, lsb_x32: int = 75) -> int:
    """round half away from zero of v / (lsb_x32/32), saturated to s16, exact rational arithmetic."""
    x = Fraction(v_uv) * 32 / lsb_x32
    mag = floor(abs(x) + Fraction(1, 2))
    c = -mag if x < 0 else mag
    return max(-32768, min(32767, c))


def trunc_div_2pow(acc: int, bits: int) -> int:
    """truncate toward zero: sign(acc) * (|acc| // 2^bits)"""
    q = abs(acc) // (1 << bits)
    return -q if acc < 0 else q


def sat24(v: int) -> int:
    return S24_MIN if v < S24_MIN else S24_MAX if v > S24_MAX else v


def biquad_cascade(codes, sos, guard_bits=10):
    """DF-I cascade. codes: list[int] of ADC codes for ONE channel. sos: list of (b0,b1,b2,a0,a1,a2) Q2.30.
    Returns (section1_out, section2_out) lists of ints."""
    stages = [[0, 0, 0, 0] for _ in sos]  # x1 x2 y1 y2
    s1, s2 = [], []
    for code in codes:
        x = sat24(code * (1 << guard_bits))
        outs = []
        for k, (b0, b1, b2, _a0, a1, a2) in enumerate(sos):
            x1, x2, y1, y2 = stages[k]
            acc = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y = sat24(trunc_div_2pow(acc, 30))
            stages[k] = [x, x1, y, y1]
            outs.append(y)
            x = y
        s1.append(outs[0]); s2.append(outs[1])
    return s1, s2


def median_floor(values):
    s = sorted(values)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return floor(Fraction(s[n // 2 - 1] + s[n // 2], 2))


def car_ref(frame, active, mode):
    vals = [v for v, a in zip(frame, active) if a]
    n = len(vals)
    if mode == "none":
        return 0
    if mode == "median":
        return median_floor(vals)
    if mode == "mean":
        recip = floor(Fraction(1 << 32, n) + Fraction(1, 2))  # round(2^32/n)
        return floor(Fraction(sum(vals) * recip + (1 << 31), 1 << 32))
    raise ValueError(mode)


def car_mean_exact_round_half_up(frame, active):
    vals = [v for v, a in zip(frame, active) if a]
    return floor(Fraction(sum(vals), len(vals)) + Fraction(1, 2))


def calibrate(x, active=True, thr_mult_q8=1024, iterations=3, mad_q16=97164, clip_q16=582983):
    """Per-channel calibration from a list of CAR outputs. Returns dict(baseline, mad, sigma_q16, thr, recip, armed)."""
    b = median_floor(x)
    m = median_floor([abs(v - b) for v in x])
    for _ in range(iterations):
        lim = (m * clip_q16) // (1 << 16)
        kept = [v for v in x if abs(v - b) <= lim]
        if len(kept) < 32:
            kept = list(x)
        b = median_floor(kept)
        m = median_floor([abs(v - b) for v in kept])
    sigma_q16 = m * mad_q16
    off = floor(Fraction(thr_mult_q8 * sigma_q16, 1 << 24) + Fraction(1, 2))
    armed = bool(active) and m > 0
    thr = max(S24_MIN + 1, min(S24_MAX, b - off)) if armed else NEVER
    recip = (1 << 48) // max(sigma_q16, 1) if armed else 0
    return dict(baseline=b, mad=m, sigma_q16=sigma_q16, thr=thr, recip=recip, armed=armed)


def amp_q8(delta, recip):
    return min(0xFFFF, (delta * recip) // (1 << 24))


def detect(trace, ts, thr, base, recip, peak_search=15, refractory=30):
    """Direct transcription of the spec rules on (value, timestamp) lists. Returns [(ts, code, delta, amp)]."""
    if thr == NEVER:
        return []
    out = []
    last_peak = -refractory - 1
    n = len(trace)
    for j in range(1, n):
        if not (trace[j - 1] >= thr and trace[j] < thr):
            continue
        if ts[j] - last_peak < refractory:
            continue
        best = j
        k = j
        while k < n and ts[k] <= ts[j] + peak_search:
            if trace[k] < trace[best]:
                best = k
            k += 1
        if ts[best] - last_peak < refractory:
            continue
        code = trace[best]
        out.append((ts[best], code, base - code, amp_q8(base - code, recip)))
        last_peak = ts[best]
    return out


def veto(events, n_active, window=6, frac_q8=128):
    """events: list of (ts, ch, code, delta, amp). Returns kept list."""
    if not events or frac_q8 <= 0:
        return list(events)
    need = max(2, -(-(frac_q8 * n_active) // 256))  # ceil
    ev = sorted(events, key=lambda e: (e[0], e[1]))
    kept, i = [], 0
    while i < len(ev):
        j = i + 1
        while j < len(ev) and ev[j][0] - ev[i][0] <= window:
            j += 1
        grp = ev[i:j]
        if len({e[1] for e in grp}) < need:
            kept.extend(grp)
        i = j
    return kept


def merge(events, window=15):
    ev = sorted(events, key=lambda e: (e[0], e[1]))
    out, i = [], 0
    while i < len(ev):
        j = i + 1
        while j < len(ev) and ev[j][0] - ev[i][0] <= window:
            j += 1
        grp = ev[i:j]
        best = grp[0]
        for e in grp[1:]:
            if e[4] > best[4]:
                best = e
        out.append(best)
        i = j
    return out


def pipeline(codes_rows, sos, active, valid, spec):
    """codes_rows: list of frames (list[int]) ; valid: list[bool]. Returns merged events [(ts, ch, code, delta, amp)]."""
    n_ch = len(codes_rows[0])
    kept_idx = [t for t, v in enumerate(valid) if v]
    cols = [[row[c] for t, row in enumerate(codes_rows) if valid[t]] for c in range(n_ch)]
    filt = [biquad_cascade(cols[c], sos, spec["guard_bits"])[1] for c in range(n_ch)]
    nt = len(kept_idx)
    ref = [car_ref([filt[c][t] for c in range(n_ch)], active, spec["car"]) for t in range(nt)]
    car = [[sat24(filt[c][t] - ref[t]) for t in range(nt)] for c in range(n_ch)]
    cands = []
    for c in range(n_ch):
        if not active[c]:
            continue
        cal = calibrate(car[c], True, spec["threshold_mult_q8"], spec["mad_iterations"])
        for (t, code, delta, amp) in detect(car[c], kept_idx, cal["thr"], cal["baseline"], cal["recip"],
                                           spec["peak_search"], spec["refractory"]):
            cands.append((t, c, code, delta, amp))
    cands.sort(key=lambda e: (e[0], e[1]))
    kept = veto(cands, sum(1 for a in active if a), spec["veto_window"], spec["veto_fraction_q8"])
    return merge(kept, spec["merge_window"]), cands
