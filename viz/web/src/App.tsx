import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import { control, eventSocket, fetchSession } from "./api";
import { BrainScene, type Pulse } from "./BrainScene";
import { KIND_COLORS, KIND_LABELS, type EventKind, type NeuralEvent, type Session } from "./types";

const ALL_KINDS = Object.keys(KIND_LABELS) as EventKind[];

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [pulses, setPulses] = useState<Pulse[]>([]);
  const [inspected, setInspected] = useState<NeuralEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showTraces, setShowTraces] = useState(true);
  const clockRef = useRef(0);

  useEffect(() => {
    fetchSession().then(setSession).catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    const socket = eventSocket();
    socket.onmessage = (message) => {
      const payload = JSON.parse(message.data) as {
        type: string;
        session?: Session;
        batch?: { events: NeuralEvent[] };
      };
      if (payload.session) setSession(payload.session);
      if (payload.batch?.events?.length) {
        const born = performance.now() / 1000;
        setPulses((current) => {
          const next = current.concat(payload.batch!.events.map((event) => ({ event, born })));
          return next.slice(-1800);
        });
        setInspected((current) => current ?? payload.batch!.events.at(-1) ?? null);
      }
    };
    socket.onerror = () => setError("WebSocket connection failed. Is the Python server running?");
    return () => socket.close();
  }, []);

  useEffect(() => {
    let frame = 0;
    const tick = (t: number) => {
      clockRef.current = t / 1000;
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  const timeline = useMemo(() => pulses.slice(-300), [pulses]);
  const tMax = Math.max(session?.t_s ?? 1, 1);

  if (!session) {
    return <div className="app"><div className="topbar">{error ?? "Connecting to Neurosig visualization server…"}</div></div>;
  }

  const send = async (body: Record<string, unknown>) => {
    const next = await control(body);
    setSession(next);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">Neurosig · live neural view</div>
        <div className="banner">{session.disclaimer}</div>
        <div>{error ?? `t = ${session.t_s.toFixed(3)} s`}</div>
      </header>

      <aside className="panel">
        <h2>Controls</h2>
        <div className="row">
          <button className={session.running ? "active" : ""} onClick={() => send({ running: true })}>Start</button>
          <button onClick={() => send({ running: false })}>Pause</button>
          <button onClick={() => send({ reset: true, seed: session.seed })}>Reset</button>
        </div>
        <label>Mode</label>
        <select value={session.mode} onChange={(event) => send({ mode: event.target.value })}>
          <option value="synthetic">Synthetic</option>
          <option value="neuropixels">Neuropixels (sorted spikes)</option>
          <option value="hybrid">Hybrid</option>
        </select>
        <label>Speed {session.speed.toFixed(2)}×</label>
        <input type="range" min={0.1} max={8} step={0.1} value={session.speed} onChange={(event) => send({ speed: Number(event.target.value) })} />
        <label>Firing rate scale {session.rate_scale.toFixed(2)}×</label>
        <input type="range" min={0.2} max={4} step={0.1} value={session.rate_scale} onChange={(event) => send({ rate_scale: Number(event.target.value) })} />
        <label>Seed</label>
        <input type="number" value={session.seed} onChange={(event) => send({ reset: true, seed: Number(event.target.value) })} />
        <label>Region</label>
        <select
          value={session.selected_region ?? ""}
          onChange={(event) => send({ selected_region: event.target.value || null })}
        >
          <option value="">All regions</option>
          {session.regions.map((region) => (
            <option key={region.region_id} value={region.region_id}>{region.label}</option>
          ))}
        </select>
        <label>Recording channel</label>
        <select
          value={session.selected_channel ?? ""}
          onChange={(event) => send({ selected_channel: event.target.value === "" ? null : Number(event.target.value) })}
        >
          <option value="">All channels</option>
          {session.electrodes.sites.map((site) => (
            <option key={site.channel_id} value={site.channel_id}>{site.label}</option>
          ))}
        </select>
        <div className="toggle">
          <input
            type="checkbox"
            checked={session.run_golden_model}
            onChange={(event) => send({ run_golden_model: event.target.checked })}
          />
          Golden-model detections
        </div>
        <div className="toggle">
          <input type="checkbox" checked={showTraces} onChange={(event) => setShowTraces(event.target.checked)} />
          Travel traces (stylized)
        </div>
        <h2>Event visibility</h2>
        {ALL_KINDS.map((kind) => (
          <label className="toggle" key={kind}>
            <input
              type="checkbox"
              checked={session.visible_kinds.includes(kind)}
              onChange={(event) => {
                const next = event.target.checked
                  ? [...session.visible_kinds, kind]
                  : session.visible_kinds.filter((item) => item !== kind);
                void send({ visible_kinds: next });
              }}
            />
            <span className="swatch" style={{ background: KIND_COLORS[kind] }} />
            {KIND_LABELS[kind]}
          </label>
        ))}
        <p className="inspect">{session.honesty.synthetic}</p>
        <p className="inspect">
          Travel traces follow stylized white-matter guides (callosal, thalamic, visual radiation). They are not measured axonal conduction.
        </p>
      </aside>

      <div className="stage">
        <Canvas camera={{ position: [150, 28, 125], fov: 38, near: 0.1, far: 2000 }}>
          <BrainScene
            session={session}
            pulses={pulses}
            showTraces={showTraces}
            onSelectRegion={(id) => void send({ selected_region: id })}
          />
        </Canvas>
      </div>

      <aside className="metrics">
        <h2>Metrics</h2>
        <div className="stat"><span>Simulated events</span><strong>{session.metrics.simulated_event_count}</strong></div>
        <div className="stat"><span>Recorded spikes</span><strong>{session.metrics.recorded_spike_count}</strong></div>
        <div className="stat"><span>Detections</span><strong>{session.metrics.detection_count}</strong></div>
        <div className="stat"><span>Missed (GT)</span><strong>{session.metrics.missed_count}</strong></div>
        <div className="stat"><span>Precision</span><strong>{session.metrics.precision == null ? "n/a" : session.metrics.precision.toFixed(2)}</strong></div>
        <div className="stat"><span>Recall</span><strong>{session.metrics.recall == null ? "n/a" : session.metrics.recall.toFixed(2)}</strong></div>
        <div className="stat"><span>Active channels</span><strong>{session.metrics.active_channels}</strong></div>
        <div className="stat"><span>Localization</span><strong>{session.localization.reason}</strong></div>
        <p className="inspect">{session.detector_note}</p>
        <p className="inspect">Neuropixels playback: {session.neuropixels.playback_kind}. Raw voltage: {String(session.neuropixels.has_raw_voltage)}.</p>
        <h2>Inspect</h2>
        {inspected ? (
          <div className="inspect">
            <div><code>{inspected.event_id}</code></div>
            <div>t = {inspected.t_s.toFixed(4)} s · {KIND_LABELS[inspected.kind]}</div>
            <div>region {inspected.region_id}</div>
            <div>xyz_mm [{inspected.xyz_mm.map((v) => v.toFixed(1)).join(", ")}]</div>
            <div>quality {inspected.position_quality}</div>
            <div>amp {inspected.amplitude.toFixed(1)} {inspected.amplitude_units}</div>
            <div>source {inspected.source_id}</div>
            <div>channel {inspected.channel_id ?? "—"}</div>
            <div>duration {inspected.duration_ms} ms</div>
          </div>
        ) : (
          <p className="inspect">Pause to keep an event in view, then inspect the latest pulse.</p>
        )}
      </aside>

      <footer className="timeline">
        <div className="clock">Timeline · click to seek · session {session.t_s.toFixed(2)} s</div>
        <div
          className="timeline-track"
          onClick={(event) => {
            const rect = event.currentTarget.getBoundingClientRect();
            const ratio = (event.clientX - rect.left) / rect.width;
            void send({ seek_s: ratio * tMax, reset: false });
          }}
        >
          {timeline.map((pulse) => (
            <div
              key={pulse.event.event_id + pulse.born}
              className="tick"
              style={{
                left: `${(pulse.event.t_s / tMax) * 100}%`,
                background: KIND_COLORS[pulse.event.kind],
              }}
              onClick={(event) => {
                event.stopPropagation();
                setInspected(pulse.event);
              }}
            />
          ))}
        </div>
      </footer>
    </div>
  );
}
