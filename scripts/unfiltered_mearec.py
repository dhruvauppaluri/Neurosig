"""Regenerate the smoke recording without MEArec's recording band-pass filter."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import MEArec as mr
import numpy as np
import yaml


def main():
    source = ROOT / "data" / "smoke"
    output = ROOT / "data" / "unfiltered"
    output.mkdir(parents=True, exist_ok=True)
    with (source / "parameters.yaml").open() as f:
        params = yaml.safe_load(f)
    params["recordings"]["recordings"]["filter"] = False
    recording = mr.gen_recordings(
        params=params["recordings"], templates=source / "templates.h5",
        template_ids=[0], n_jobs=1, tmp_mode=None,
    )
    path = output / "recording.h5"
    mr.save_recording_generator(recording, filename=path)
    (output / "parameters.yaml").write_text(yaml.safe_dump(params))

    # Check the persisted data, including the ground truth, before plotting it.
    loaded = mr.load_recordings(path)
    original = mr.load_recordings(source / "recording.h5")
    samples = np.asarray(loaded.recordings)
    assert loaded.info["recordings"]["filter"] == False
    assert samples.shape == original.recordings.shape == (32000, 32)
    assert np.isfinite(samples).all()
    assert not np.array_equal(samples, original.recordings)
    assert len(loaded.spiketrains) == len(original.spiketrains)
    for actual, expected in zip(loaded.spiketrains, original.spiketrains):
        np.testing.assert_array_equal(actual.times, expected.times)

    time_s = np.arange(samples.shape[0]) / float(loaded.info["recordings"]["fs"])
    fig, axes = plt.subplots(8, 4, figsize=(16, 16), sharex=True, sharey=True)
    for channel, ax in enumerate(axes.flat):
        ax.plot(time_s, samples[:, channel], linewidth=0.35, color="#166185")
        ax.set_title(f"Channel {channel}", fontsize=10)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8, labelleft=True)
    fig.suptitle("Unfiltered MEArec recording · 32 channels · 32 kHz · seed 42\n"
                 "Shared voltage scale across channels; recording band-pass disabled", fontsize=14)
    fig.supxlabel("Time (s)")
    fig.supylabel("Electrode voltage (µV)")
    fig.tight_layout(rect=(0.025, 0.025, 1, 0.95))
    fig.savefig(output / "traces.png", dpi=140)
    plt.close(fig)
    print(f"Validated unfiltered recording and unchanged spike times: {path}")
    print(f"Plot: {output / 'traces.png'}")


if __name__ == "__main__":
    main()
