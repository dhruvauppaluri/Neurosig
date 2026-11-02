"""Apply the floating-point causal IIR spike-band reference to ADC codes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class IirConfig:
    sample_rate_hz: float = 32_000.0
    highpass_hz: float = 300.0
    lowpass_hz: float = 6_000.0
    order_per_section: int = 2

    def __post_init__(self) -> None:
        nyquist = self.sample_rate_hz / 2.0
        if not 0 < self.highpass_hz < self.lowpass_hz < nyquist:
            raise ValueError("cutoffs must satisfy 0 < highpass < lowpass < Nyquist")
        if self.order_per_section != 2:
            raise ValueError("this hardware reference currently uses second-order sections")


def design_sos(config: IirConfig) -> np.ndarray:
    """Create a high-pass biquad followed by a low-pass biquad."""
    highpass = signal.butter(
        config.order_per_section,
        config.highpass_hz,
        btype="highpass",
        fs=config.sample_rate_hz,
        output="sos",
    )
    lowpass = signal.butter(
        config.order_per_section,
        config.lowpass_hz,
        btype="lowpass",
        fs=config.sample_rate_hz,
        output="sos",
    )
    return np.vstack((highpass, lowpass))


def section_poles(sos: np.ndarray) -> list[np.ndarray]:
    """Return each biquad's poles; stability requires every magnitude to be below one."""
    return [np.roots(section[3:]) for section in np.asarray(sos)]


def maximum_pole_magnitude(sos: np.ndarray) -> float:
    return max(float(np.max(np.abs(poles))) for poles in section_poles(sos))


def filter_adc_codes(codes: np.ndarray, lsb_uv: float, sos: np.ndarray) -> np.ndarray:
    """Reconstruct microvolts and filter each channel with zero initial state."""
    samples = np.asarray(codes)
    if samples.ndim != 2:
        raise ValueError(f"expected [samples, channels], got {samples.shape}")
    if lsb_uv <= 0:
        raise ValueError("lsb_uv must be positive")
    reconstructed_uv = samples.astype(np.float64) * lsb_uv
    return signal.sosfilt(sos, reconstructed_uv, axis=0)


def response_at(sos: np.ndarray, frequencies_hz: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    _, response = signal.sosfreqz(sos, worN=frequencies_hz, fs=sample_rate_hz)
    return 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))


def save_plots(
    output: Path,
    raw_uv: np.ndarray,
    filtered_uv: np.ndarray,
    sos: np.ndarray,
    config: IirConfig,
) -> None:
    frequencies, response = signal.sosfreqz(sos, worN=8192, fs=config.sample_rate_hz)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(frequencies[1:], 20 * np.log10(np.maximum(np.abs(response[1:]), 1e-12)))
    ax.axvline(config.highpass_hz, color="tab:orange", linestyle="--", label="300 Hz")
    ax.axvline(config.lowpass_hz, color="tab:green", linestyle="--", label="6 kHz")
    ax.set(xlabel="Frequency (Hz)", ylabel="Gain (dB)", title="Causal IIR spike-band response", ylim=(-80, 5))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "frequency_response.png", dpi=150)
    plt.close(fig)

    channel = 21
    start = int(0.11 * config.sample_rate_hz)
    stop = int(0.15 * config.sample_rate_hz)
    time_ms = np.arange(start, stop) / config.sample_rate_hz * 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time_ms, raw_uv[start:stop, channel], linewidth=0.8)
    axes[0].set(ylabel="Voltage (µV)", title="Channel 21: reconstructed ADC input")
    axes[1].plot(time_ms, filtered_uv[start:stop, channel], linewidth=0.8, color="tab:red")
    axes[1].set(xlabel="Time (ms)", ylabel="Voltage (µV)", title="Channel 21: causal IIR output")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "channel21_example.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", type=Path, default=Path("data/quantized/adc_codes.npy"))
    parser.add_argument("--adc-summary", type=Path, default=Path("data/quantized/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("data/filtered"))
    parser.add_argument("--sample-rate", type=float, default=32_000.0)
    parser.add_argument("--highpass", type=float, default=300.0)
    parser.add_argument("--lowpass", type=float, default=6_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = IirConfig(args.sample_rate, args.highpass, args.lowpass)
    codes = np.load(args.codes)
    adc_summary = json.loads(args.adc_summary.read_text())
    lsb_uv = float(adc_summary["adc"]["lsb_uv"])
    sos = design_sos(config)

    max_pole = maximum_pole_magnitude(sos)
    if max_pole >= 1.0:
        raise RuntimeError(f"unstable floating-point filter: maximum pole magnitude {max_pole}")

    filtered_uv = filter_adc_codes(codes, lsb_uv, sos)
    if not np.isfinite(filtered_uv).all():
        raise RuntimeError("filter output contains NaN or infinite samples")
    raw_uv = codes.astype(np.float64) * lsb_uv

    probe_frequencies = np.array([100.0, 300.0, 1_000.0, 6_000.0, 10_000.0])
    gains_db = response_at(sos, probe_frequencies, config.sample_rate_hz)
    poles = section_poles(sos)
    summary = {
        "input_codes": str(args.codes),
        "shape": list(filtered_uv.shape),
        "filter": {
            **asdict(config),
            "architecture": "cascade: second-order Butterworth high-pass, then second-order Butterworth low-pass",
            "implementation": "causal second-order sections with zero state after reset",
            "sos_coefficients": sos.tolist(),
        },
        "stability": {
            "criterion": "all pole magnitudes must be less than 1",
            "maximum_pole_magnitude": max_pole,
            "stable": True,
            "poles_by_section": [
                [{"real": float(p.real), "imag": float(p.imag), "magnitude": float(abs(p))} for p in pair]
                for pair in poles
            ],
        },
        "gain_db": {str(int(freq)): float(gain) for freq, gain in zip(probe_frequencies, gains_db)},
        "input_range_uv": [float(np.min(raw_uv)), float(np.max(raw_uv))],
        "output_range_uv": [float(np.min(filtered_uv)), float(np.max(filtered_uv))],
        "note": "Float64 reference only; fixed-point coefficient and state stability remain to be verified.",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "filtered_uv.npy", filtered_uv)
    np.save(args.output / "sos_coefficients.npy", sos)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_plots(args.output, raw_uv, filtered_uv, sos, config)
    print(json.dumps(summary, indent=2))
    print(f"Wrote floating-point IIR reference to {args.output.resolve()}")


if __name__ == "__main__":
    main()
