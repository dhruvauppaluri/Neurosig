from __future__ import annotations

import unittest

import numpy as np

from edge import SAMPLE_RATE_HZ, SEED
from edge.coords import euclidean_um
from edge.forward import PointSource, render_recording
from edge.geometry import linear_probe_um
from edge.localize import AmplitudeLocalizer, BeamformLocalizer, LocalizationEngine, RobustScanLocalizer, TemplateLocalizer
from edge.localize.common import aligned_trough_amplitudes
from edge.localize.template import default_training_xyz_um
from edge.snippets import extract_snippet
from edge.types import Snippet, snippet_sample_counts

# Low-noise single-source amplitude error measured on this geometry (seed 20261103):
# keep a realistic cap; the method is expected well under 40 µm when the 1/r fit
# recovers the lateral offset. If a future change regresses, raise this only after
# measuring, and document the new bound here.
AMPLITUDE_LOW_NOISE_MAX_ERROR_UM = 40.0
HELD_OUT = (30.0, 300.0, 0.0)


def _snippet_for(
    xyz_um: tuple[float, float, float],
    *,
    noise_uv: float,
    seed: int = SEED,
    extra_sources: list[PointSource] | None = None,
) -> Snippet:
    array = linear_probe_um()
    n_pre, n_post = snippet_sample_counts(SAMPLE_RATE_HZ)
    t_peak = n_pre + 30
    sources = [PointSource(t_sample=t_peak, xyz_um=xyz_um, amplitude_uv=12.0)]
    if extra_sources:
        sources.extend(extra_sources)
    voltages = render_recording(
        sources,
        array,
        SAMPLE_RATE_HZ,
        noise_uv=noise_uv,
        seed=seed,
        n_samples=t_peak + n_post + 40,
    )
    return extract_snippet(
        voltages,
        t_peak,
        array,
        ground_truth_xyz_um=xyz_um,
    )


class LocalizeProtocolTests(unittest.TestCase):
    def test_engines_match_protocol(self) -> None:
        array = linear_probe_um()
        engines = [
            AmplitudeLocalizer(),
            TemplateLocalizer(array=array, training_xyz_um=default_training_xyz_um()),
            BeamformLocalizer(array=array),
            RobustScanLocalizer(array=array),
        ]
        for engine in engines:
            self.assertIsInstance(engine, LocalizationEngine)


class AmplitudeKnownLocationTests(unittest.TestCase):
    def test_low_noise_single_source_error_bound(self) -> None:
        snippet = _snippet_for(HELD_OUT, noise_uv=0.5, seed=SEED)
        estimate = AmplitudeLocalizer().estimate(snippet)
        self.assertTrue(estimate.available)
        self.assertEqual(estimate.method, "amplitude")
        self.assertIsNotNone(estimate.xyz_um)
        self.assertIsNotNone(estimate.error_um)
        self.assertGreaterEqual(estimate.latency_s, 0.0)
        # error_um is micrometers via edge.coords.euclidean_um, not millimetres.
        self.assertAlmostEqual(estimate.error_um, euclidean_um(estimate.xyz_um, HELD_OUT), places=9)
        self.assertLess(
            estimate.error_um,
            AMPLITUDE_LOW_NOISE_MAX_ERROR_UM,
            msg=f"measured amplitude error {estimate.error_um:.3f} µm exceeds bound",
        )


class HeldOutTemplateTests(unittest.TestCase):
    def test_training_catalog_excludes_test_location(self) -> None:
        training = default_training_xyz_um()
        distances = np.linalg.norm(training - np.asarray(HELD_OUT), axis=1)
        self.assertGreater(float(np.min(distances)), 1.0)

    def test_template_localizer_on_held_out_site(self) -> None:
        array = linear_probe_um()
        training = default_training_xyz_um()
        self.assertFalse(any(np.allclose(row, HELD_OUT) for row in training))
        snippet = _snippet_for(HELD_OUT, noise_uv=0.5, seed=SEED)
        estimate = TemplateLocalizer(array=array, training_xyz_um=training).estimate(snippet)
        self.assertTrue(estimate.available)
        self.assertEqual(estimate.method, "template")
        self.assertIsNotNone(estimate.error_um)
        self.assertLess(estimate.error_um, 40.0)


class NoiseAndOverlapTests(unittest.TestCase):
    def test_noise_sweep_returns_available_estimates(self) -> None:
        array = linear_probe_um()
        engines = [
            AmplitudeLocalizer(),
            TemplateLocalizer(array=array),
            BeamformLocalizer(array=array),
            RobustScanLocalizer(array=array),
        ]
        for noise in (1.0, 8.0, 40.0, 80.0):
            snippet = _snippet_for(HELD_OUT, noise_uv=noise, seed=SEED + int(noise))
            for engine in engines:
                estimate = engine.estimate(snippet)
                if engine.method == "robust_scan" and noise >= 80.0:
                    continue
                self.assertTrue(estimate.available, msg=f"{engine.method} failed at noise={noise}")
                self.assertIsNotNone(estimate.xyz_um)
                self.assertGreaterEqual(estimate.latency_s, 0.0)
                # Do not hardcode fake accuracy; only require a finite micrometer error.
                self.assertTrue(np.isfinite(estimate.error_um))
                self.assertGreaterEqual(estimate.error_um, 0.0)

    def test_two_simultaneous_sources_robust_scan_fails_closed(self) -> None:
        array = linear_probe_um()
        n_pre, _n_post = snippet_sample_counts(SAMPLE_RATE_HZ)
        t_peak = n_pre + 30
        extra = [
            PointSource(t_sample=t_peak, xyz_um=(40.0, 420.0, 10.0), amplitude_uv=12.0),
        ]
        snippet = _snippet_for(HELD_OUT, noise_uv=8.0, extra_sources=extra)
        engines = [
            AmplitudeLocalizer(),
            TemplateLocalizer(array=array),
            BeamformLocalizer(array=array),
            RobustScanLocalizer(array=array),
        ]
        for engine in engines:
            estimate = engine.estimate(snippet)
            if engine.method == "robust_scan":
                self.assertFalse(estimate.available, msg="robust_scan must fail closed on two sources")
                self.assertIsNone(estimate.xyz_um)
                self.assertIn("bimodal", estimate.reason)
                continue
            self.assertTrue(estimate.available)
            self.assertIsNotNone(estimate.xyz_um)
            self.assertTrue(np.isfinite(estimate.error_um))


class CoordinateUnitTests(unittest.TestCase):
    def test_error_is_micrometers_not_millimetres(self) -> None:
        # 20 µm along the shank is 20.0 in this frame, not 0.02 mm.
        self.assertAlmostEqual(euclidean_um((0.0, 0.0, 0.0), (0.0, 20.0, 0.0)), 20.0)
        snippet = _snippet_for((20.0, 200.0, 0.0), noise_uv=0.5)
        estimate = BeamformLocalizer().estimate(snippet)
        gt = snippet.ground_truth_xyz_um
        self.assertAlmostEqual(estimate.error_um, euclidean_um(estimate.xyz_um, gt), places=9)
        self.assertLess(estimate.error_um, 200.0)
        # Recording a 20 µm contact pitch as 20.0 (not 0.02) is the unit check.


class AlignedAmplitudeTests(unittest.TestCase):
    def test_aligned_trough_ignores_distant_channel_noise_minima(self) -> None:
        array = linear_probe_um()
        n_pre, n_post = snippet_sample_counts(SAMPLE_RATE_HZ)
        n = n_pre + n_post
        voltages = np.zeros((n, array.n_channels))
        voltages[n_pre, 10] = -12.0
        voltages[2, 0] = -30.0  # earlier noise spike on a far contact
        snippet = Snippet(
            event_id=0,
            t_sample=n_pre,
            t0_sample=0,
            sample_rate_hz=SAMPLE_RATE_HZ,
            channel_ids=array.channel_ids(),
            voltages_uv=voltages,
            electrode_xyz_um=array.xyz_um(),
            peak_channel=10,
            ground_truth_xyz_um=HELD_OUT,
        )
        aligned = aligned_trough_amplitudes(snippet)
        self.assertEqual(int(np.argmax(aligned)), 10)
        estimate = RobustScanLocalizer(array=array).estimate(_snippet_for(HELD_OUT, noise_uv=0.5))
        self.assertTrue(estimate.available)
        self.assertLess(estimate.error_um, 80.0)


if __name__ == "__main__":
    unittest.main()
