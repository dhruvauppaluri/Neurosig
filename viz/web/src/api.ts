import type { Session } from "./types";

export interface SnippetPayload {
  event_id: string;
  sample_rate_hz: number;
  channel_ids: number[];
  t0_sample: number;
  t_sample: number;
  voltages_uv: number[][];
}

export async function fetchSession(): Promise<Session> {
  const response = await fetch("/api/v1/session");
  if (!response.ok) throw new Error("session fetch failed");
  return response.json();
}

export async function control(body: Record<string, unknown>): Promise<Session> {
  const response = await fetch("/api/v1/session/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error("control failed");
  return response.json();
}

export async function fetchSnippet(eventId: string): Promise<SnippetPayload | null> {
  const response = await fetch(`/api/v1/snippet/${encodeURIComponent(eventId)}`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error("snippet fetch failed");
  return response.json() as Promise<SnippetPayload>;
}

export function eventSocket(): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${protocol}://${window.location.host}/api/v1/events`);
}
