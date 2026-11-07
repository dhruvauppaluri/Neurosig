"""Phase 8 production localizer: K=1/2 space-time fit, calibrated confidence regions, accept/reject.

Calibration (``Calibration``) is fit on the dev pool only. Mondrian split-conformal scaling gives per-stratum
squared-Mahalanobis quantiles so nominal 90% regions reach ~90% empirical coverage on held-out events.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import chi2

from edge.localize.common import fold_linear_probe_ambiguity
from edge.phase8.multisource import SpaceTimeFitter
from edge.types import ElectrodeArray, LocalizationEstimate, Snippet

SCHEMA_VERSION = "phase8-result-1"


@dataclass
class Calibration:
    tau_two: float = 150.0          # delta above which a second source is declared
    tau_gray: float = 120.0         # delta in (tau_gray, tau_two] flags a possible unresolved overlap
    q_clean: float = 4.6            # conformal squared-Mahalanobis quantile (nominal level) per stratum
    q_flagged: float = 4.6
    q_two: float = 4.6
    tau_shared: float = 15.0        # shared-waveform delta above which a second source is declared
    q_shared: float = 4.6
    nominal: float = 0.90
    max_unit_area_um2: float = 1e9  # reject if pi*sqrt(det cov) (region area at q=1) exceeds this
    max_chi2_red: float = 3.0       # reject if single-source fit residual is anomalous
    min_sep_um: float = 30.0        # two sources closer than this are merged to one

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class SourceEstimate:
    xyz_um: tuple[float, float, float]
    cov_xy: list
    region_area_um2: float
    energy: float


@dataclass
class LocalizationResult:
    event_id: int | None
    sources: list[SourceEstimate]
    n_sources: int
    confidence: float
    accepted: bool
    reject_reason: str
    flags: list = field(default_factory=list)
    delta: float = 0.0
    chi2_red: float = 0.0
    runtime_ms: float = 0.0
    schema: str = SCHEMA_VERSION

    @property
    def primary(self) -> SourceEstimate | None:
        return self.sources[0] if self.sources else None

    def to_dict(self) -> dict:
        p = self.primary
        return {
            "schema": self.schema, "event_id": self.event_id,
            "available": bool(self.accepted and p is not None), "method": "phase8_spacetime",
            "xyz_um": None if p is None else list(p.xyz_um), "confidence": self.confidence,
            "accepted": self.accepted, "reason": self.reject_reason, "flags": list(self.flags),
            "n_sources": self.n_sources,
            "sources": [{"xyz_um": list(s.xyz_um), "cov_xy_um2": s.cov_xy, "region_area_um2": s.region_area_um2,
                         "energy": s.energy} for s in self.sources],
            "delta": self.delta, "chi2_red": self.chi2_red, "latency_s": self.runtime_ms / 1000.0,
            "unverified": True,
        }


def _ellipse_area(cov: np.ndarray, q: float) -> float:
    det = float(np.linalg.det(cov))
    return float(np.pi * q * np.sqrt(max(det, 0.0)))


class Phase8Localizer:
    method = "phase8_spacetime"

    def __init__(self, array: ElectrodeArray, calibration: Calibration | None = None, **fitter_kw) -> None:
        self.array = array
        fitter_kw.setdefault("rank", 4)
        self.fitter = SpaceTimeFitter(array, **fitter_kw)
        self.cal = calibration or Calibration()

    # raw features reused by the calibrator
    def analyze(self, snippet: Snippet, mask: np.ndarray | None = None):
        Wt, sigma = self.fitter.whiten(snippet, mask)
        fit = self.fitter.fit(snippet, mask=mask)
        return fit, Wt, sigma

    def localize(self, snippet: Snippet, analysis=None, mask: np.ndarray | None = None) -> LocalizationResult:
        t0 = time.perf_counter()
        fit, Wt, sigma = analysis or self.analyze(snippet, mask)
        if fit is None:
            return LocalizationResult(snippet.event_id, [], 0, 0.0, False, "zero signal", ["no_signal"],
                                      runtime_ms=(time.perf_counter() - t0) * 1000)
        cal = self.cal
        flags: list[str] = []
        two_free = fit.delta > cal.tau_two
        two_shared = (not two_free) and fit.delta_shared > cal.tau_shared
        two = two_free or two_shared
        srcs = fit.sources2 if two_free else (fit.sources_shared if two_shared else fit.sources1)
        if two and np.hypot(srcs[0].x - srcs[1].x, srcs[0].y - srcs[1].y) < cal.min_sep_um:
            two, srcs = False, fit.sources1
            flags.append("merged_close_pair")
        f = self.fitter
        if two and any(sx.x >= f.hi[0] - 1.0 or sx.y >= f.hi[1] - 1.0 or sx.y <= f.lo[1] + 1.0 for sx in srcs):
            two, two_free, two_shared, srcs = False, False, False, fit.sources1
            flags.append("boundary_pair_rejected")
        pos = np.array([[s.x, s.y] for s in srcs])
        cov_all = self.fitter.covariance(pos, Wt, sigma)
        n_obs = Wt.size
        chi2_red = fit.sse2 / max(n_obs - 2 * pos.shape[0] - pos.shape[0] * Wt.shape[1], 1) if two else \
            fit.sse1 / max(n_obs - 2 - Wt.shape[1], 1)
        if (not two) and fit.delta > cal.tau_gray:
            flags.append("possible_unresolved_overlap")
        order = np.argsort([-s.energy for s in srcs])
        out: list[SourceEstimate] = []
        q = cal.q_shared if two_shared else (cal.q_two if two else (cal.q_flagged if flags else cal.q_clean))
        if two_shared:
            flags.append("shared_waveform_pair")
        for k in order:
            c = cov_all[2 * k:2 * k + 2, 2 * k:2 * k + 2]
            c = 0.5 * (c + c.T) + 1e-9 * np.eye(2)
            s = srcs[k]
            out.append(SourceEstimate((s.x, s.y, 0.0), c.tolist(), _ellipse_area(c, q), s.energy))
        primary_area = out[0].region_area_um2
        unit_area = _ellipse_area(np.asarray(out[0].cov_xy), 1.0)
        reason = ""
        if chi2_red > cal.max_chi2_red:
            reason = f"fit residual chi2_red={chi2_red:.2f} exceeds {cal.max_chi2_red}"
        elif unit_area > cal.max_unit_area_um2:
            reason = f"uncertainty area {unit_area:.0f} um^2 exceeds {cal.max_unit_area_um2:.0f}"
        conf = float(np.clip(1.0 - unit_area / max(min(cal.max_unit_area_um2, 1e6), 1.0), 0.0, 1.0))
        return LocalizationResult(snippet.event_id, out, len(out), conf, reason == "", reason, flags,
                                  float(fit.delta), float(chi2_red), (time.perf_counter() - t0) * 1000)

    def estimate(self, snippet: Snippet) -> LocalizationEstimate:
        r = self.localize(snippet)
        p = r.primary
        gt = snippet.ground_truth_xyz_um
        if p is None or not r.accepted:
            return LocalizationEstimate(False, self.method, None, gt, None, r.runtime_ms / 1000, 0.0,
                                        r.reject_reason, snippet.event_id)
        err = None
        if gt is not None:
            a = np.asarray(fold_linear_probe_ambiguity(p.xyz_um, snippet.electrode_xyz_um))
            b = np.asarray(fold_linear_probe_ambiguity(gt, snippet.electrode_xyz_um))
            err = float(np.linalg.norm(a - b))
        return LocalizationEstimate(True, self.method, p.xyz_um, gt, err, r.runtime_ms / 1000, r.confidence,
                                    "calibrated space-time fit", snippet.event_id)


def fit_calibration(array: ElectrodeArray, events, base: Calibration | None = None,
                    false_two_target: float = 0.03, gray_pct: float = 90.0, accept_target: float = 0.92,
                    false_shared_target: float = 0.01) -> Calibration:
    """Calibrate thresholds and conformal quantiles on dev events only. Returns a new Calibration."""
    cal = Calibration(**(base.to_json() if base else {}))
    loc = Phase8Localizer(array, cal)
    analyses = [loc.analyze(e.snippet) for e in events]
    single = np.array([not e.contaminated for e in events])
    deltas = np.array([a[0].delta for a in analyses])
    cal.tau_two = float(np.percentile(deltas[single], 100 * (1 - false_two_target)))
    cal.tau_gray = float(np.percentile(deltas[single], gray_pct))
    d_sh = np.array([a[0].delta_shared for a in analyses])
    cal.tau_shared = float(np.percentile(d_sh[single], 100 * (1 - false_shared_target)))
    cal.max_chi2_red = 1e9
    loc.cal = cal
    chis = []
    for e, an in zip(events, analyses):
        if not e.contaminated:
            chis.append(loc.localize(e.snippet, an).chi2_red)
    cal.max_chi2_red = float(1.1 * np.percentile(chis, 98))
    loc.cal = cal
    unit_areas = []
    for e, an in zip(events, analyses):
        r = loc.localize(e.snippet, an)
        if r.primary is not None and r.reject_reason == "" or (r.primary is not None and "chi2_red" not in r.reject_reason):
            unit_areas.append(_ellipse_area(np.asarray(r.primary.cov_xy), 1.0))
    cal.max_unit_area_um2 = float(np.percentile(unit_areas, 100 * accept_target))
    loc.cal = cal
    # conformal scores (accepted events only, so accepted-event coverage is calibrated) per stratum on the primary source, Mahalanobis in folded xy
    scores = {"clean": [], "flagged": [], "two": [], "shared": []}
    for e, an in zip(events, analyses):
        r = loc.localize(e.snippet, an)
        if r.primary is None or not r.accepted:
            continue
        truth = np.array([np.hypot(e.truth_xyz_um[0], e.truth_xyz_um[2]), e.truth_xyz_um[1]])
        d = np.array(r.primary.xyz_um[:2]) - truth
        m = float(d @ np.linalg.solve(np.array(r.primary.cov_xy), d))
        key = ("shared" if "shared_waveform_pair" in r.flags else "two") if r.n_sources == 2 else \
            ("flagged" if r.flags else "clean")
        scores[key].append(m)

    def q(vals):
        if len(vals) < 10:
            return float(chi2.ppf(cal.nominal, 2) * 4)
        k = int(np.ceil((len(vals) + 1) * cal.nominal))
        return float(np.sort(vals)[min(k, len(vals)) - 1])
    cal.q_clean, cal.q_flagged, cal.q_two = q(scores["clean"]), q(scores["flagged"]), q(scores["two"])
    cal.q_shared = q(scores["shared"])
    cal.__dict__["_n_scores"] = {k: len(v) for k, v in scores.items()}
    return cal


def save_calibration(cal: Calibration, path: str | Path) -> None:
    d = cal.to_json()
    Path(path).write_text(json.dumps(d, indent=2) + "\n")
