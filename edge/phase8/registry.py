"""Append-only experiment registry and shared error statistics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

REGISTRY_PATH = Path("reports/phase8/experiment_registry.jsonl")


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_revision() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def versions() -> dict:
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "git": git_revision()}


def log_experiment(entry: dict, path: Path = REGISTRY_PATH) -> None:
    """Required keys: id, hypothesis, split, params, before, after, decision, reasoning."""
    required = {"id", "hypothesis", "split", "params", "before", "after", "decision", "reasoning"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"registry entry missing {sorted(missing)}")
    record = dict(entry)
    record["config_hash"] = config_hash(entry["params"])
    record["versions"] = versions()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=float) + "\n")


def error_stats(errors) -> dict:
    """Distribution of finite errors (um). Rejected events are counted by the caller."""
    e = np.asarray([x for x in errors if x is not None and np.isfinite(x)], dtype=float)
    if e.size == 0:
        return {"n": 0}
    return {
        "n": int(e.size),
        "mean": float(e.mean()),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "worst": float(e.max()),
    }


def bootstrap_median_diff(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap of median(a-b) over matched events; positive means b is better."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return {"n": int(a.size), "median_diff": float(np.median(a) - np.median(b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
