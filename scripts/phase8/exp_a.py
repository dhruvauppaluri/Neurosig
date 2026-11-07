"""Goal A development experiments on the dev pool. Usage: .venv/bin/python scripts/phase8/exp_a.py"""

from edge.benchmark_v3 import _array, _engines
from edge.localize import get_localizer
from edge.phase8.evaluate import fmt, run_engines, summarize
from edge.phase8.refine import RefinedScanLocalizer
from edge.phase8.synth import make_events

SUBSETS = {"single": lambda e: not e.contaminated, "contam": lambda e: e.contaminated,
           "noise2": lambda e: e.noise_uv == 2.0, "noise10": lambda e: e.noise_uv == 10.0,
           "noise25": lambda e: e.noise_uv == 25.0}

if __name__ == "__main__":
    array = _array("linear")
    events = make_events("dev", n_per_cell=40)
    engines = _engines(array)
    engines["refined"] = RefinedScanLocalizer(array)
    engines["refined_nosvd"] = RefinedScanLocalizer(array, use_temporal_svd=False)
    engines["refined_norefine"] = RefinedScanLocalizer(array, refine=False)
    res = run_engines(engines, events)
    table = summarize(res, events, SUBSETS)
    for s in ("all", "single", "contam", "noise2", "noise10", "noise25"):
        print(f"--- {s}")
        print(fmt(table, s))
