"""Phase 12 FINAL-pool biophysical templates (fresh seed 20271308, same 6 cell models as Phase 11B; used only by the Phase 12 final evaluation): MEArec/NEURON templates from 6 BBP L5 cell models on the
Neuropixels-128 (1.0 checkerboard) probe with known soma positions. Seeded (20271308). Output (git-ignored):
data/phase12/biophys_final/templates.h5. None of NeuroSig's forward models is involved in generating these templates."""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

MODELS = ["L5_TTPC1_cADpyr232_1", "L5_UTPC_cADpyr232_1", "L5_STPC_cADpyr232_1", "L5_LBC_bAC217_1",
          "L5_NBC_bAC217_1", "L5_MC_bAC217_1"]
N_PER_MODEL = 12


def simulate(work, output):
    import MEArec as mr
    import yaml
    package = Path(mr.__file__).parent
    models = work / "models"
    for m in MODELS:
        shutil.copytree(package / "cell_models" / "bbp" / m, models / m)
    with (package / "default_params" / "templates_params.yaml").open() as f:
        p = yaml.safe_load(f)
    p.update(n=N_PER_MODEL, seed=20271308, probe="Neuropixels-128", ncontacts=1, timeout=300, xlim=[10, 80],
             rot="physrot", min_amp=30)
    t = mr.gen_templates(models, params=p, templates_tmp_folder=work / "templates", parallel=False, n_jobs=1,
                         verbose=False)
    output.mkdir(parents=True, exist_ok=True)
    mr.save_template_generator(t, filename=output / "templates.h5")
    (output / "parameters.yaml").write_text(yaml.safe_dump(p))
    print("templates", t.templates.shape, flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "data" / "phase12" / "biophys_final"
    with tempfile.TemporaryDirectory(prefix="neurosig-") as folder:
        work = Path(folder)
        (work / "venv").symlink_to(root / ".venv", target_is_directory=True)
        worker = work / "gen.py"; shutil.copy2(__file__, worker)
        wrappers = work / "bin"; wrappers.mkdir()
        c = wrappers / "nrnivmodl"
        c.write_text(f'#!/bin/sh\nexec "{work / "venv/bin/python"}" "{work / "venv/bin/nrnivmodl"}" "$@"\n'); c.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join([str(wrappers), str(work / "venv" / "bin"), env.get("PATH", "")])
        env["MPLBACKEND"] = "Agg"; env["MPLCONFIGDIR"] = str(root / ".cache" / "matplotlib")
        subprocess.run([str(work / "venv" / "bin" / "python"), str(worker), "--worker", str(work), str(output)],
                       cwd=work, env=env, check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        simulate(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
