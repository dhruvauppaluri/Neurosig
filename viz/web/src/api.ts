import type { Session } from "./types";

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

export function eventSocket(): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${protocol}://${window.location.host}/api/v1/events`);
}
