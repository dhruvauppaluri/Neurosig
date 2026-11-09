"""Phase 14 MEArec/NEURON template sets, reserved BEFORE any Phase 14 evaluation.

Morphology-disjoint design. Phases 11-12 used 6 BBP L5 cell models (TTPC1, UTPC, STPC, LBC, NBC, MC). The 7 cell
models never simulated in NeuroSig are reserved as the Phase 14 FINAL morphologies:

  final_np1   7 unseen models x 12 positions, Neuropixels-128 (1.0 checkerboard), seed 20271411
  final_np2   7 unseen models x 8 positions, Neuropixels2-128 (2.0 two-column linear), seed 20271412
  sel_np2     the 3 Phase 14 'select' models (UTPC, NBC, STPC) x 8, Neuropixels2-128, seed 20271413 (geometry
              transfer is selected on these, never on final_np2)

Usage: .venv/bin/python scripts/phase14/gen_templates14.py <set>. Output (git-ignored): data/phase14/templates_<set>.h5.
Refuses to overwrite. No NeuroSig forward model is involved.
"""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

UNSEEN = ["L5_TTPC2_cADpyr232_1", "L5_BP_bAC217_1", "L5_BTC_bAC217_1", "L5_ChC_cACint209_1", "L5_DBC_bAC217_1",
          "L5_NGC_bNAC219_1", "L5_SBC_bNAC219_1"]
SETS = {
    "final_np1": dict(models=UNSEEN, n=12, seed=20271411, probe="Neuropixels-128"),
    "final_np2": dict(models=UNSEEN, n=8, seed=20271412, probe="Neuropixels2-128"),
    "sel_np2": dict(models=["L5_UTPC_cADpyr232_1", "L5_NBC_bAC217_1", "L5_STPC_cADpyr232_1"], n=8, seed=20271413,
                    probe="Neuropixels2-128"),
}


def simulate(work, name, output):
    import MEArec as mr
    import yaml
    cfg = SETS[name]
    package = Path(mr.__file__).parent
    models = work / "models"
    for m in cfg["models"]:
        shutil.copytree(package / "cell_models" / "bbp" / m, models / m)
    with (package / "default_params" / "templates_params.yaml").open() as f:
        p = yaml.safe_load(f)
    p.update(n=cfg["n"], seed=cfg["seed"], probe=cfg["probe"], ncontacts=1, timeout=300, xlim=[10, 80],
             rot="physrot", min_amp=30)
    t = mr.gen_templates(models, params=p, templates_tmp_folder=work / "templates", parallel=False, n_jobs=1,
                         verbose=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    mr.save_template_generator(t, filename=output)
    output.with_suffix(".yaml").write_text(yaml.safe_dump({**p, "models": cfg["models"]}))
    print("templates", t.templates.shape, flush=True)


def main(name):
    root = Path(__file__).resolve().parents[2]
    output = root / "data" / "phase14" / f"templates_{name}.h5"
    if output.exists():
        raise SystemExit(f"{output} exists; reserved template sets are never regenerated")
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
        subprocess.run([str(work / "venv" / "bin" / "python"), str(worker), "--worker", str(work), name, str(output)],
                       cwd=work, env=env, check=True)


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        simulate(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
    else:
        main(sys.argv[1])
