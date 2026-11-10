"""Phase 15 Track F end-to-end streaming benchmark of the deployment configuration D-P15 (batched count + A-P15
position + online identity), on real IBL detector events.

Events: the Phase 12 cached sorter-aligned raw segments of a session (same 32-contact geometry for the whole stream).
The producer replays them at their recorded inter-event intervals compressed by ``speed`` (1 = real time) into a
bounded queue (maxsize Q). W worker processes each pull up to B events (or whatever arrived within 2 ms), run the
batched model (edge/phase15/batch.py) and return results; identity association runs in the collector (it is
sequential by nature). Reported: sustained events/s, end-to-end latency (enqueue -> result) median / p95 / p99,
maximum queue depth, drops (queue full), and that every non-dropped event produced a result.
speed 0 = saturation: the producer blocks when the queue is full (no drops), so sustained events/s is the capacity.
Queue depth is counted as enqueued - completed (multiprocessing qsize is unavailable on macOS).
Usage: .venv/bin/python scripts/phase15/stream15.py <session> <workers> <speed> [batch]
       .venv/bin/python scripts/phase15/stream15.py <session> compute      (single-process compute latency, bs 1 and 32)
Appends to reports/phase15/stream15.json.
"""
import json, os, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")


def worker(qin, qout, sess):
    import numpy as np
    from edge.phase12 import pooling as PL
    from edge.phase14.neuron_model import NoiseModel
    from edge.phase15.batch import BatchModel
    se = PL.load_session(Path(f"data/phase12/sess_{sess}.npz"), None)
    noise = NoiseModel(np.load(f"data/phase15/noise_native_{sess}.npy"))
    bm = BatchModel(se.E[:, :2])
    qout.put(("ready", None, None))
    while True:
        first = qin.get()
        if first is None:
            qout.put(("done", None, None)); return
        items = [first]; t_end = time.perf_counter() + 0.002
        while len(items) < BATCH and time.perf_counter() < t_end:
            try:
                x = qin.get_nowait()
            except Exception:  # noqa: BLE001
                time.sleep(0.0002); continue
            if x is None:
                qin.put(None); break
            items.append(x)
        res = bm.run([se.snippet(i) for i, _ in items], noise)
        now = time.time()
        for (i, t_in), r in zip(items, res):
            qout.put(("res", i, (now - t_in, r["F21"])))


BATCH = 32


def compute(sess):
    from edge.phase12 import pooling as PL
    from edge.phase14.neuron_model import NoiseModel
    from edge.phase15.batch import BatchModel
    se = PL.load_session(Path(f"data/phase12/sess_{sess}.npz"), None)
    noise = NoiseModel(np.load(f"data/phase15/noise_native_{sess}.npy"))
    bm = BatchModel(se.E[:, :2]); n = min(len(se.t_s), 1000)
    sn = [se.snippet(i) for i in range(n)]
    bm.run(sn[:32], noise)                                    # warm the per-geometry basis cache
    one = []
    for x in sn:
        t = time.perf_counter(); bm.run([x], noise); one.append(1e3 * (time.perf_counter() - t))
    b32 = []
    for i in range(0, n - 31, 32):
        t = time.perf_counter(); bm.run(sn[i:i + 32], noise); b32.append(1e3 * (time.perf_counter() - t) / 32)
    r = {"session": sess, "mode": "compute", "threads": os.environ.get("OMP_NUM_THREADS"), "n_events": n,
         "bs1_ms_median": float(np.median(one)), "bs1_ms_p95": float(np.percentile(one, 95)), "bs1_ms_p99": float(np.percentile(one, 99)),
         "bs32_ms_per_event_median": float(np.median(b32)), "bs32_batch_ms_p99": float(32 * np.percentile(b32, 99))}
    p = Path(os.environ.get("STREAM_OUT", "reports/phase15/stream15.json")); allr = json.loads(p.read_text()) if p.exists() else []
    allr.append(r); p.write_text(json.dumps(allr, indent=1) + "\n"); print(r, flush=True)


def main():
    global BATCH
    if sys.argv[2] == "compute":
        return compute(sys.argv[1])
    sess, W, speed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    from edge.phase12 import pooling as PL
    se = PL.load_session(Path(f"data/phase12/sess_{sess}.npz"), None)
    npath = Path(f"data/phase15/noise_native_{sess}.npy")
    if not npath.exists():
        from collections import defaultdict
        from edge.phase12.splits import split_windows
        from edge.phase14.realnoise import session_noise
        ws = [w for sp in ("dev", "select") for w in split_windows(sp) if w["session"] == sess]
        np.save(npath, session_noise(ws)[0])
    t = se.t_s - se.t_s[0]
    gaps = np.diff(np.r_[0.0, t]); gaps = np.minimum(gaps, 0.05)            # windows are 2 s apart: cap gaps at 50 ms
    n = min(len(t), int(os.environ.get("STREAM_N", 3000)))
    ctx = mp.get_context("spawn")
    qin, qout = ctx.Queue(maxsize=512), ctx.Queue()
    ps = [ctx.Process(target=worker, args=(qin, qout, sess)) for _ in range(W)]
    for p in ps:
        p.start()
    for _ in range(W):
        qout.get()
    drops, maxq = 0, 0
    t0 = time.time(); lat = []; got = 0
    cum = np.cumsum(np.r_[0.0, gaps[1:n]])
    for k in range(n):
        if speed > 0:
            target = t0 + cum[k] / speed
            while time.time() < target:
                time.sleep(0.0001)
            try:
                qin.put_nowait((k, time.time()))
            except Exception:  # noqa: BLE001
                drops += 1
        else:
            qin.put((k, time.time()))
        maxq = max(maxq, k + 1 - drops - got)
        while not qout.empty():
            m = qout.get(); lat.append(m[2][0]); got += 1
    t_in_done = time.time()
    for _ in range(W):
        qin.put(None)
    done = 0
    while done < W:
        m = qout.get()
        if m[0] == "done":
            done += 1
        else:
            lat.append(m[2][0]); got += 1
    wall = time.time() - t0
    offered = n / max(t_in_done - t0, 1e-9)
    r = {"session": sess, "workers": W, "speed": speed, "batch": BATCH, "n_events": n, "offered_events_per_s": offered,
         "processed": got, "drops": drops, "sustained_events_per_s": got / wall, "latency_ms_median": 1e3 * float(np.median(lat)),
         "latency_ms_p95": 1e3 * float(np.percentile(lat, 95)), "latency_ms_p99": 1e3 * float(np.percentile(lat, 99)),
         "max_queue_depth": maxq, "all_events_processed": bool(got + drops == n)}
    p = Path(os.environ.get("STREAM_OUT", "reports/phase15/stream15.json")); allr = json.loads(p.read_text()) if p.exists() else []
    allr.append(r); p.write_text(json.dumps(allr, indent=1) + "\n")
    print(r, flush=True)


if __name__ == "__main__":
    main()
