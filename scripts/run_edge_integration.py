"""Compare fresh RTL events to Python, then exercise the software NSG1 edge link.

Requires /tmp/neurosig-rtl-events.txt from the Verilator command in README.
NSG1 frames are produced by AsicModel in Python; there is no RTL packetizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.asic_model import AsicModel
from edge.pipeline import EventAssembler, decode_windows, localize_events
from scripts.detect_spikes import DetectorConfig
from scripts.fixed_point_model import detect_integer_candidates, merge_integer_events
from scripts.quantize_adc import AdcConfig


def run(observed: Path, recording: Path, filtered: Path, thresholds: Path) -> dict:
    rtl = [tuple(map(int, line.split())) for line in observed.read_text().splitlines() if line.strip()]
    codes = np.load(filtered)
    expected = merge_integer_events(
        detect_integer_candidates(codes, np.load(thresholds), DetectorConfig()), DetectorConfig()
    )
    python_events = [(e.sample, e.channel, e.code) for e in expected]
    if rtl != python_events:
        raise AssertionError(f"RTL events differ from Python: {rtl} != {python_events}")
    with h5py.File(recording) as handle:
        voltages = np.asarray(handle["recordings"], dtype=np.float64)
        positions = np.asarray(handle["channel_positions"], dtype=np.float64)
    if voltages.shape != codes.shape or positions.shape != (codes.shape[1], 3):
        raise ValueError("MEArec recording and fixed-point vectors disagree")
    # MEArec channel_positions are in micrometers; keep the original geometry.
    geometry = {i: tuple(float(v) for v in positions[i]) for i in range(len(positions))}
    model = AsicModel(n_channels=codes.shape[1])
    assembler = EventAssembler(session_id="mearec-seed42", electrode_positions_um=geometry)
    packet_count = 0
    for index, (sample, channel, code) in enumerate(rtl, 1):
        # Filter output code belongs to RTL; raw unfiltered voltage supplies snippet.
        event = type("RtlEvent", (), {"sample": sample, "channel": channel,
                                      "voltage_uv": code * AdcConfig().lsb_uv})()
        frames = model.emit(event, voltages, event_id=index)
        packet_count += len(frames)
        for frame in reversed(frames):
            assembler.feed(frame)
    events = assembler.finish()
    if len(events) != len(rtl) or any(e.snippet is None for e in events):
        raise AssertionError("packet assembly changed spike count or lost waveform")
    for edge_event, (sample, channel, _code) in zip(events, rtl):
        if (edge_event.t_sample, edge_event.channel_id) != (sample, channel):
            raise AssertionError("edge event differs from RTL timestamp/channel")
    localized = localize_events(events)
    # Existing 32-feature RTL model is a count-only comparison, not a trained
    # classifier for this one-neuron recording.
    weights_file = Path("data/decoder/weights.txt")
    scores = []
    if weights_file.exists():
        params = np.loadtxt(weights_file, dtype=np.int64)
        predictions = decode_windows(localized, params[1:].astype(float) / 256.0,
                                     float(params[0]) / 256.0)
        scores = [p.score for p in predictions]
    return {
        "input": str(recording), "rtl_trace": str(observed),
        "rtl_python_exact_events": len(rtl), "software_nsg1_packets": packet_count,
        "edge_logical_events": len(events), "with_waveforms": sum(e.snippet is not None for e in events),
        "with_localization": sum(e.localization is not None and e.localization.available for e in localized),
        "count_decoder_scores": scores,
        "ingest_diagnostics": vars(assembler.diagnostics),
        "source_note": "RTL event triples from fresh Verilator trace; NSG1 packet bytes produced by Python AsicModel",
        "accuracy_note": "Count decoder is trained on a different synthetic task; scores here have no accuracy interpretation. MEArec positions are not established neuron-localization truth.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, default=Path("/tmp/neurosig-rtl-events.txt"))
    parser.add_argument("--recording", type=Path, default=Path("data/unfiltered/recording.h5"))
    parser.add_argument("--filtered", type=Path, default=Path("data/fixed_point/filtered_codes_i24.npy"))
    parser.add_argument("--thresholds", type=Path, default=Path("data/fixed_point/threshold_codes.npy"))
    parser.add_argument("--report", type=Path, default=Path("reports/edge-integration.json"))
    args = parser.parse_args()
    report = run(args.observed, args.recording, args.filtered, args.thresholds)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RTL/Python/edge events: {report['rtl_python_exact_events']}/{report['edge_logical_events']}")


if __name__ == "__main__":
    main()
