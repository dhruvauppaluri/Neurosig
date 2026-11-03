import { useEffect, useMemo, useState } from "react";
import { fetchSnippet, type SnippetPayload } from "./api";
import type { NeuralEvent } from "./types";

function shouldLoadSnippet(event: NeuralEvent | null): boolean {
  if (!event) return false;
  return event.kind === "detected_spike" || event.kind === "localization_estimate";
}

function polyline(values: number[], width: number, height: number, vmin: number, vmax: number): string {
  if (!values.length) return "";
  const span = Math.max(1e-6, vmax - vmin);
  return values
    .map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * width;
      const y = height - ((value - vmin) / span) * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

export function WaveformPanel({ event }: { event: NeuralEvent | null }) {
  const [snippet, setSnippet] = useState<SnippetPayload | null>(null);
  const [status, setStatus] = useState<string>("");

  useEffect(() => {
    if (!shouldLoadSnippet(event)) {
      setSnippet(null);
      setStatus("");
      return;
    }
    let cancelled = false;
    setStatus("Loading snippet…");
    void fetchSnippet(event!.event_id).then((payload) => {
      if (cancelled) return;
      setSnippet(payload);
      setStatus(payload ? "" : "No cached snippet for this event.");
    });
    return () => {
      cancelled = true;
    };
  }, [event?.event_id, event?.kind]);

  const traces = useMemo(() => {
    if (!snippet) return [];
    const samples = snippet.voltages_uv;
    if (!samples.length) return [];
    const nChannels = snippet.channel_ids.length;
    const channels: number[][] = Array.from({ length: nChannels }, () => []);
    for (const row of samples) {
      for (let channel = 0; channel < nChannels; channel += 1) {
        channels[channel].push(Number(row[channel] ?? 0));
      }
    }
    let vmin = 0;
    let vmax = 0;
    for (const trace of channels) {
      for (const value of trace) {
        vmin = Math.min(vmin, value);
        vmax = Math.max(vmax, value);
      }
    }
    if (vmin === vmax) {
      vmin -= 1;
      vmax += 1;
    }
    return snippet.channel_ids.map((channelId, index) => ({
      channelId,
      values: channels[index],
      vmin,
      vmax,
      peak: event?.channel_id === channelId,
    }));
  }, [snippet, event?.channel_id]);

  if (!shouldLoadSnippet(event)) return null;

  return (
    <div className="waveforms">
      <div className="wave-caption">Channel snippets (µV)</div>
      {status && <div className="wave-status">{status}</div>}
      {traces.map((trace) => (
        <div className={trace.peak ? "wave-row peak" : "wave-row"} key={trace.channelId}>
          <span>ch{String(trace.channelId).padStart(2, "0")}</span>
          <svg viewBox="0 0 160 18" preserveAspectRatio="none" aria-hidden="true">
            <polyline
              fill="none"
              stroke={trace.peak ? "#80ffdb" : "#7ea0b5"}
              strokeWidth="1"
              points={polyline(trace.values, 160, 18, trace.vmin, trace.vmax)}
            />
          </svg>
        </div>
      ))}
    </div>
  );
}
