"""F-series: conditioner options on top of baseline subtraction, real + null events, development windows."""
import json, sys, time
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.evalset import build_events_shift, evaluate, table
from edge.phase9.factories import baseline_factory
from edge.phase9.improved import conditioned_factory
S="/private/tmp/claude-501/-Users-dhruvauppaluri--claude-spares-spare-zfwvDI/ab536295-60fa-4914-8ddc-e384323aba86/scratchpad/"
wins = H.manifest_windows("dev")
real, null = build_events_shift(wins, 0.0), build_events_shift(wins, 0.020)
B = dict(baseline=True)
variants = {"raw(legacy)": baseline_factory,
            "B": conditioned_factory(baseline_factory, **B),
            "B+car": conditioned_factory(baseline_factory, car=True, **B),
            "B+tmean3": conditioned_factory(baseline_factory, tmean=True, **B),
            "B+tproj": conditioned_factory(baseline_factory, tproj=True, **B),
            "B+noise_norm": conditioned_factory(baseline_factory, noise_norm=True, **B),
            "B+interp_dead": conditioned_factory(baseline_factory, interp_dead=True, **B),
            "B+tmedian3": conditioned_factory(baseline_factory, tmedian=True, **B)}
res = {}
for k, f in variants.items():
    t = time.time(); res[k] = evaluate(f, real, null); print(k, round(time.time()-t), "s", flush=True)
json.dump(res, open(S+"fseries.json", "w"), default=float)
print(table(res))
