import json, sys, time
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.factories import baseline_factory
S="/private/tmp/claude-501/-Users-dhruvauppaluri--claude-spares-spare-zfwvDI/ab536295-60fa-4914-8ddc-e384323aba86/scratchpad/"
ev = H.build_events(H.manifest_windows("dev"), conditioning="raw")
t=time.time(); r = H.run(ev, baseline_factory); print("run", round(time.time()-t), "s")
json.dump(r, open(S+"base_raw_dev.json","w"), default=float)
keys=[k for k in next(iter(r.values())) if k.startswith(("drop_","noise_"))]
print("method".ljust(14), *[k.replace("drop_","d_").replace("noise_","n_")[:14].ljust(14) for k in keys])
for m,v in r.items():
    print(m.ljust(14), *[f"{v[k]['median']:5.1f}/{v[k]['p90']:5.0f}/{100*v[k]['failure_fraction']:2.0f}%".ljust(14) if v[k].get('n') else 'na'.ljust(14) for k in keys])
