"""Generate one biophysical neuron and a seeded one-second, 32-channel recording."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def simulate(work, output):
    import importlib.metadata
    import MEArec as mr
    import numpy as np
    import yaml

    package = Path(mr.__file__).parent
    models = work / "models"
    model = "L5_STPC_cADpyr232_1"
    shutil.copytree(package / "cell_models" / "bbp" / model, models / model)
    with (package / "default_params" / "templates_params.yaml").open() as f:
        template_params = yaml.safe_load(f)
    template_params.update(n=2, seed=42, probe="Neuronexus-32", ncontacts=1, timeout=120)
    templates = mr.gen_templates(
        models, params=template_params, templates_tmp_folder=work / "templates",
        parallel=False, n_jobs=1, verbose=True,
    )
    if templates.templates.shape[0] < 1:
        raise RuntimeError("Biophysical template generation produced no templates")
    with (package / "default_params" / "recordings_params.yaml").open() as f:
        recording_params = yaml.safe_load(f)
    recording_params["spiketrains"].update(duration=1, n_exc=1, n_inh=0, f_exc=10, st_exc=0)
    recording_params["seeds"] = dict(spiketrains=42, templates=42, convolution=42, noise=42)
    recording = mr.gen_recordings(
        params=recording_params, tempgen=templates, template_ids=[0],
        n_jobs=1, tmp_mode=None, verbose=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    mr.save_template_generator(templates, filename=output / "templates.h5")
    mr.save_recording_generator(recording, filename=output / "recording.h5")
    loaded = mr.load_recordings(output / "recording.h5")
    samples = np.asarray(loaded.recordings)
    assert samples.shape == (32000, 32), samples.shape
    assert np.isfinite(samples).all()
    assert np.std(samples) > 0
    spike_count = sum(len(train) for train in loaded.spiketrains)
    assert spike_count > 0
    summary = {
        "versions": {name: importlib.metadata.version(name) for name in ("MEArec", "NEURON", "LFPy", "numpy")},
        "python": sys.version.split()[0], "seed": 42, "probe": "Neuronexus-32",
        "cell_model": model, "duration_s": 1, "sample_rate_hz": 32000,
        "recording_shape": list(samples.shape), "dtype": str(samples.dtype),
        "units": "microvolts", "template_count": int(len(templates.templates)),
        "neuron_count": len(loaded.spiketrains), "ground_truth_spikes": spike_count,
        "validation": "HDF5 reload, expected shape, finite nonconstant samples, nonempty ground truth",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "parameters.yaml").write_text(yaml.safe_dump({"templates": template_params, "recordings": recording_params}))
    print(json.dumps(summary, indent=2))


def main():
    # MEArec invokes unquoted shell commands internally. A temporary alias keeps
    # Python/package paths free of spaces even when this checkout has spaces.
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "smoke"
    with tempfile.TemporaryDirectory(prefix="neurosig-") as folder:
        work = Path(folder)
        (work / "venv").symlink_to(root / ".venv", target_is_directory=True)
        worker = work / "smoke_mearec.py"
        shutil.copy2(__file__, worker)
        # Installed entry points have an absolute interpreter in their shebang;
        # invoke nrnivmodl through the alias so its makefile paths stay usable.
        wrappers = work / "bin"
        wrappers.mkdir()
        compiler = wrappers / "nrnivmodl"
        compiler.write_text(
            f'#!/bin/sh\nexec "{work / "venv/bin/python"}" "{work / "venv/bin/nrnivmodl"}" "$@"\n'
        )
        compiler.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join([str(wrappers), str(work / "venv" / "bin"), env.get("PATH", "")])
        env["MPLBACKEND"] = "Agg"
        env["MPLCONFIGDIR"] = str(root / ".cache" / "matplotlib")
        subprocess.run(
            [str(work / "venv" / "bin" / "python"), str(worker), "--worker", str(work), str(output)],
            cwd=work, env=env, check=True,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        simulate(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
