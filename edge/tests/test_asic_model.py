from __future__ import annotations

import unittest

import numpy as np

from edge.asic_model import AsicModel, default_lsb_uv, quantize_uv
from edge.link.codec import EventPacket, SnippetPacket, decode_packet
from edge.link.control import ControlPlane
from edge.link.schema import ACQ_BOTH, FLAG_HAS_SNIPPET
from scripts.detect_spikes import Event
from scripts.quantize_adc import AdcConfig


class AsicModelTests(unittest.TestCase):
    def test_lsb_matches_adc_config(self) -> None:
        self.assertEqual(default_lsb_uv(), AdcConfig().lsb_uv)

    def test_preserves_t_sample_as_snippet_peak_index(self) -> None:
        n_pre, n_post = 32, 48
        n_ch = 4
        t_sample = 80
        voltages = np.zeros((200, n_ch), dtype=np.float64)
        voltages[t_sample, 2] = -40.0
        voltages[t_sample - 1, 2] = -10.0
        event = Event(sample=t_sample, channel=2, voltage_uv=-40.0, normalized_amplitude=-6.0)
        model = AsicModel(
            n_channels=n_ch,
            n_pre=n_pre,
            n_post=n_post,
            control=ControlPlane(n_channels=n_ch, mode=ACQ_BOTH),
        )
        packets = model.packets_for_event(event, voltages)
        self.assertEqual(len(packets), 2)
        ev, sn = packets
        self.assertIsInstance(ev, EventPacket)
        self.assertIsInstance(sn, SnippetPacket)
        assert isinstance(ev, EventPacket)
        assert isinstance(sn, SnippetPacket)
        self.assertEqual(ev.t_sample, t_sample)
        self.assertEqual(sn.t_sample, t_sample)
        self.assertEqual(ev.t_sample, sn.t_sample)
        peak_index = sn.t_sample - sn.t0_sample
        self.assertEqual(peak_index, n_pre)
        self.assertEqual(sn.peak_index, peak_index)
        self.assertEqual(sn.peak_channel, 2)
        self.assertEqual(ev.flags & FLAG_HAS_SNIPPET, FLAG_HAS_SNIPPET)
        peak_codes = sn.samples[peak_index]
        self.assertEqual(int(peak_codes[2]), ev.amplitude)
        expected = int(quantize_uv(np.asarray([-40.0]), model.lsb_uv).reshape(-1)[0])
        self.assertEqual(ev.amplitude, expected)
        # Round-trip keeps the shared timestamp and peak row.
        restored_sn = decode_packet(model.emit(event, voltages)[1])
        assert isinstance(restored_sn, SnippetPacket)
        self.assertEqual(restored_sn.t_sample, event.sample)
        self.assertEqual(restored_sn.peak_index, n_pre)


if __name__ == "__main__":
    unittest.main()
