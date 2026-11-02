"""HTTP and WebSocket API for the visualization session."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from viz.sim.engine import NeuralEventEngine

ROOT = Path(__file__).resolve().parents[1]
WEB_DIST = ROOT / "web" / "dist"
WEB_INDEX = WEB_DIST / "index.html"

engine = NeuralEventEngine()
app = FastAPI(title="Neurosig visualization", version="1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ControlBody(BaseModel):
    mode: str | None = None
    running: bool | None = None
    reset: bool | None = None
    seed: int | None = None
    speed: float | None = None
    rate_scale: float | None = None
    visible_kinds: list[str] | None = None
    selected_region: str | None = Field(default=None)
    selected_channel: int | None = None
    run_golden_model: bool | None = None
    seek_s: float | None = None


@app.get("/api/v1/session")
def get_session() -> dict:
    return engine.snapshot()


@app.post("/api/v1/session/control")
def post_control(body: ControlBody) -> dict:
    payload = body.model_dump(exclude_unset=True)
    return engine.control(payload)


@app.get("/api/v1/timeline")
def get_timeline(limit: int = 400) -> dict:
    return {"points": engine.timeline_points(limit)}


@app.websocket("/api/v1/events")
async def event_stream(socket: WebSocket) -> None:
    await socket.accept()
    try:
        await socket.send_json({"type": "session", "session": engine.snapshot()})
        while True:
            batch = engine.step_wall()
            await socket.send_json({"type": "batch", "batch": batch.to_dict(), "session": engine.snapshot()})
            await asyncio.sleep(1.0 / 30.0)
    except WebSocketDisconnect:
        return


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_INDEX)

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        candidate = WEB_DIST / path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_INDEX)


def main() -> None:
    import uvicorn

    uvicorn.run("viz.sim.server:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
