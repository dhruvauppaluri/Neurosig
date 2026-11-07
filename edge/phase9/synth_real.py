"""Real-like synthetic events (known truth) with the measured real footprint and nuisances.

Calibration (reports/phase9/real_footprint.json, dev windows): relative amplitude vs distance follows a line source
1/sqrt(r^2+L^2) with L~20 um; spike FWHM ~0.27 ms; raw AP traces carry static per-channel DC offsets and a shared
noise component. Geometry is a real 32-contact IBL selection (4-column staggered). Truth is scored in (x, y).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edge.forward import triphasic_template
from edge.phase9 import harness as H
from edge.snippets import extract_snippet
from edge.types import ElectrodeArray, Snippet
from scripts.phase5.data import load_window, window_path

POOL_SEEDS = {"dev": 20269101, "select": 20269102, "final": 20269103}
FS = 30000.0


def reference_array() -> ElectrodeArray:
    w = H.manifest_windows("dev")[0]
    from edge.phase9.events import window_array
    return window_array(load_window(window_path(w["pid"], w["t0_s"], w["duration_s"])))


@dataclass(frozen=True)
class RealSynthEvent:
    event_id: int
    pool: str
    noise_uv: float
    amp_uv: float
    truth_xy: tuple
    sources_xy: tuple
    contaminated: bool
    snippet: Snippet
    resolvable: bool = False
    sources_full: tuple = ()
    kind: str = "single"


def _render(sources, E, noise, rng, lam_true, dc_sigma, cm_frac, n=120, kernel_ms=1.0):
    nk = int(round(kernel_ms * FS / 1000))
    k = triphasic_template(nk)
    pk = int(np.argmin(k))
    v = np.zeros((n, E.shape[0]))
    for (x, y, z, amp, t) in sources:
        r = np.sqrt((E[:, 0] - x) ** 2 + (E[:, 1] - y) ** 2 + (E[:, 2] - z) ** 2)
        g = lam_true / np.sqrt(r ** 2 + lam_true ** 2)
        s = t - pk
        lo, hi = max(s, 0), min(s + nk, n)
        v[lo:hi] += amp * g[None, :] * k[lo - s:hi - s, None]     # k is unit-peak negative: trough of depth amp*g
    v += rng.normal(0, noise, v.shape)
    v += rng.normal(0, cm_frac * noise, (n, 1))                   # shared component
    v += rng.normal(0, dc_sigma, (1, E.shape[0]))                  # static offsets
    return v


def make_events(pool: str, n_per_noise: int = 60, contaminated_fraction: float = 0.0, lam_true: float = 20.0, n_sources_choices=(2,), z_max: float = 30.0,
                array: ElectrodeArray | None = None, dc_sigma: float = 15.0, cm_frac: float = 0.5):
    array = array or reference_array()
    E = array.xyz_um()
    rng = np.random.default_rng(POOL_SEEDS[pool])
    ev = []
    for noise in (5.0, 10.0, 20.0):
        for _ in range(n_per_noise):
            x = float(rng.uniform(0, 70)); y = float(rng.uniform(E[:, 1].min() + 40, E[:, 1].max() - 40)); z = float(rng.uniform(0, z_max))
            amp = float(rng.uniform(60, 160))
            src = [(x, y, z, amp, 60)]
            truths = [(x, y)]
            cont = bool(rng.random() < contaminated_fraction)
            if cont:
                dy = float(rng.choice([-1, 1]) * rng.uniform(40, 120))
                x2, y2 = float(rng.uniform(0, 70)), float(np.clip(y + dy, E[:, 1].min() + 20, E[:, 1].max() - 20))
                src.append((x2, y2, float(rng.uniform(0, z_max)), float(amp * rng.uniform(0.5, 1.0)), 60 + int(rng.choice([-1, 1]) * rng.integers(1, 10))))
                truths.append((x2, y2))
            tr = _render(src, E, noise, rng, lam_true, dc_sigma, cm_frac)
            sn = extract_snippet(tr, 60, array, event_id=len(ev), sample_rate_hz=FS)
            ev.append(RealSynthEvent(len(ev), pool, noise, amp, truths[0], tuple(truths), cont, sn,
                                     _resolvable(src, E, noise, lam_true), tuple(src), "pair" if cont else "single"))
    return array, ev


def _resolvable(src, E, noise, lam) -> bool:
    """Predefined resolvable pair: separation >= 60 um, both peak-channel SNR >= 6, amplitude ratio >= 1/3."""
    if len(src) != 2:
        return False
    (x1, y1, z1, a1, t1), (x2, y2, z2, a2, t2) = src
    sep = float(np.hypot(x1 - x2, y1 - y2))
    def peak(x, y, z, a):
        r = np.sqrt((E[:, 0] - x) ** 2 + (E[:, 1] - y) ** 2 + (E[:, 2] - z) ** 2)
        return a * float(np.max(lam / np.sqrt(r ** 2 + lam ** 2)))
    snr = min(peak(x1, y1, z1, a1), peak(x2, y2, z2, a2)) / noise
    return bool(sep >= 60 and snr >= 6 and min(a1, a2) / max(a1, a2) >= 1 / 3)


def xy_error(est_xyz, truth_xy) -> float:
    return float(np.hypot(est_xyz[0] - truth_xy[0], est_xyz[1] - truth_xy[1]))
