from __future__ import annotations

import unittest

import numpy as np

from edge import SAMPLE_RATE_HZ, SEED
from edge.forward import PointSource, render_recording
from edge.geometry import linear_probe_um
from edge.snippets import extract_snippet
from edge.types import snippet_sample_counts


class SnippetTests(unittest.TestCase):
    def test_t_sample_minus_t0_equals_n_pre(self) -> None:
        array = linear_probe_um()
        n_pre, n_post = snippet_sample_counts(SAMPLE_RATE_HZ)
        t_sample = 200
        voltages = np.zeros((400, 32), dtype=np.float64)
        snippet = extract_snippet(voltages, t_sample, array, sample_rate_hz=SAMPLE_RATE_HZ)
        self.assertEqual(snippet.t_sample - snippet.t0_sample, n_pre)
        self.assertEqual(snippet.n_pre, n_pre)
        self.assertEqual(snippet.voltages_uv.shape, (n_pre + n_post, 32))
        self.assertEqual(snippet.electrode_xyz_um.shape, (32, 3))

    def test_peak_alignment_at_n_pre(self) -> None:
        array = linear_probe_um()
        n_pre, n_post = snippet_sample_counts(SAMPLE_RATE_HZ)
        t_sample = n_pre + 20
        source = PointSource(t_sample=t_sample, xyz_um=(30.0, 300.0, 0.0), amplitude_uv=12.0)
        voltages = render_recording(
            [source], array, SAMPLE_RATE_HZ, noise_uv=0.0, seed=SEED, n_samples=t_sample + n_post + 40
        )
        snippet = extract_snippet(
            voltages,
            t_sample,
            array,
            ground_truth_xyz_um=source.xyz_um,
        )
        self.assertEqual(snippet.t_sample - snippet.t0_sample, n_pre)
        peak_sample = int(np.argmin(np.min(snippet.voltages_uv, axis=1)))
        self.assertEqual(peak_sample, n_pre)
        self.assertIsNotNone(snippet.peak_channel)
        peak_ch = snippet.channel_ids.index(snippet.peak_channel)
        self.assertEqual(int(np.argmin(snippet.voltages_uv[:, peak_ch])), n_pre)

    def test_zero_padding_out_of_range(self) -> None:
        array = linear_probe_um()
        voltages = np.ones((10, 32), dtype=np.float64)
        snippet = extract_snippet(voltages, t_sample=2, array=array)
        n_pre, n_post = snippet_sample_counts(SAMPLE_RATE_HZ)
        self.assertEqual(snippet.voltages_uv.shape[0], n_pre + n_post)
        self.assertEqual(snippet.t0_sample, 2 - n_pre)
        self.assertTrue(np.all(snippet.voltages_uv[0 : n_pre - 2] == 0.0))


if __name__ == "__main__":
    unittest.main()
