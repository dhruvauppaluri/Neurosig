import cProfile, pstats, time
import numpy as np
from edge.benchmark_v3 import _array
from edge.phase8.pipeline import Phase8Localizer
from edge.phase8.synth import make_events
array = _array("linear"); ev = make_events("dev", n_per_cell=8)
p8 = Phase8Localizer(array)
t = time.perf_counter()
for e in ev: p8.localize(e.snippet)
print("ms/event", (time.perf_counter() - t) / len(ev) * 1000)
pr = cProfile.Profile(); pr.enable()
for e in ev[:12]: p8.localize(e.snippet)
pr.disable(); pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
