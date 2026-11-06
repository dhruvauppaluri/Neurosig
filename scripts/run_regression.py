#!/usr/bin/env python3
"""Single-command software regression: numerical, fixed-point, alignment and (when cached) real-data checks.

    .venv/bin/python scripts/run_regression.py [--out reports/regression-summary.json] [--no-realdata]

Groups (each a pytest invocation or a command), summarised in one machine-readable JSON. Status vocabulary:
  passed / failed / skipped / error      per test
  missing_data (group)                   every test in the group skipped for a missing dataset
  missing_dataset                        a skip whose reason starts with 'missing dataset' (never counted as a pass)
A group is 'pass' only when it ran at least one test and had no failure or error. Overall exit code is non-zero on any
failure or error; skipped/missing-dataset tests do not fail the run but are listed so they cannot be mistaken for passes.
No FPGA hardware or network access is needed; large real-recording windows are optional.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
GROUPS = {
    "python_dsp": ["tests/test_detect_spikes.py", "tests/test_filter_iir.py", "tests/test_quantize_adc.py", "tests/test_real_detector.py", "tests/test_fixed_point_model.py"],
    "phase5_detector_and_fixed": ["tests/test_phase5_detectors.py", "tests/test_phase5_fixed.py"],
    "fixed_point_arithmetic_oracles": ["tests/test_phase6_arithmetic.py"],
    "rtl_test_vectors": ["tests/test_phase6_vectors.py"],
    "timestamp_alignment": ["tests/test_phase6_sync.py"],
    "edge_package": ["edge/tests"],
}
REALDATA = ["tests/test_phase6_realdata.py"]
COMMANDS = {"vector_manifest_check": [PY, "scripts/phase6/generate_rtl_vectors.py", "--check"]}


def run_pytest(name, paths):
    xml = ROOT / ".cache" / f"junit_{name}.xml"
    xml.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    p = subprocess.run([PY, "-m", "pytest", *paths, "-q", "-rs", f"--junitxml={xml}", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    tests, skipped = [], []
    if xml.exists():
        for case in ET.parse(xml).getroot().iter("testcase"):
            ident = f"{case.get('classname')}::{case.get('name')}"
            status, reason = "passed", None
            for child in case:
                if child.tag in ("failure", "error"):
                    status = "failed" if child.tag == "failure" else "error"; reason = (child.get("message") or "")[:200]
                elif child.tag == "skipped":
                    status = "skipped"; reason = child.get("message") or ""
            if status == "skipped" and reason.lower().startswith("missing dataset"):
                status = "missing_dataset"
            tests.append({"id": ident, "status": status, **({"reason": reason} if reason else {})})
    counts = {k: sum(t["status"] == k for t in tests) for k in ("passed", "failed", "error", "skipped", "missing_dataset")}
    ok = p.returncode in (0, 5) and counts["failed"] == 0 and counts["error"] == 0 and counts["passed"] > 0
    return {"kind": "pytest", "paths": paths, "returncode": p.returncode, "seconds": round(time.time() - t, 1), "counts": counts,
            "status": "pass" if ok else ("not_run" if not tests else ("missing_data" if counts["passed"] == 0 and counts["failed"] == 0 and counts["error"] == 0 and counts["missing_dataset"] > 0 else "fail")), "not_passing": [t for t in tests if t["status"] != "passed"],
            "stdout_tail": "" if ok else p.stdout[-1500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "regression-summary.json")
    ap.add_argument("--no-realdata", action="store_true")
    a = ap.parse_args()
    res = {"python": sys.version.split()[0], "groups": {}}
    for n, paths in GROUPS.items():
        res["groups"][n] = run_pytest(n, paths)
    if not a.no_realdata:
        res["groups"]["real_data_regression"] = run_pytest("real_data_regression", REALDATA)
    for n, cmd in COMMANDS.items():
        t = time.time(); p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        res["groups"][n] = {"kind": "command", "cmd": " ".join(cmd[1:]), "returncode": p.returncode, "seconds": round(time.time() - t, 1),
                            "status": "pass" if p.returncode == 0 else "fail", "stdout_tail": p.stdout[-300:]}
    tot = {k: sum(g.get("counts", {}).get(k, 0) for g in res["groups"].values()) for k in ("passed", "failed", "error", "skipped", "missing_dataset")}
    res["totals"] = tot
    res["overall"] = "fail" if (tot["failed"] or tot["error"] or any(g["status"] == "fail" for g in res["groups"].values())) else \
                     ("pass_with_missing_data" if tot["missing_dataset"] or tot["skipped"] else "pass")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    for n, g in res["groups"].items():
        c = g.get("counts", {})
        print(f"{n:34s} {g['status']:5s} {c if c else ''}")
    print("TOTAL", tot, "->", res["overall"], "| summary:", a.out)
    return 1 if res["overall"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
