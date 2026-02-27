import asyncio
import json
import threading
from typing import Any, Dict, List, Optional

import yaml
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from shared.mqtt_client import MqttClient

app = FastAPI()

_latest_state: Optional[Dict[str, Any]] = None
_subscribers: List[asyncio.Queue] = []
_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@app.on_event("startup")
async def startup():
    """Capture the running event loop so the MQTT thread can post into it."""
    global _loop
    _loop = asyncio.get_event_loop()


# ── existing endpoints ────────────────────────────────────────────────────────

@app.get("/status")
def health():
    return {"status": "ok"}


@app.get("/state")
def state():
    with _lock:
        return _latest_state if _latest_state is not None else {"status": "no_state_yet"}


# ── new: real-time SSE stream ─────────────────────────────────────────────────

@app.get("/stream")
async def stream(request: Request):
    """
    Server-Sent Events endpoint.
    Connect once and receive every baby-state update as it arrives.

    Each event looks like:
        data: {"ts":1234567890,"source":"fusion_service","data":{...}}\n\n

    Usage examples:
        curl http://<pi-ip>:8000/stream
        EventSource("http://<pi-ip>:8000/stream")   // browser / JS
        URLSession with text/event-stream            // Swift / iOS
    """
    queue: asyncio.Queue = asyncio.Queue()

    with _lock:
        _subscribers.append(queue)
        snapshot = _latest_state  # send whatever we already have immediately

    async def event_generator():
        try:
            # Push the current state the moment the client connects
            if snapshot is not None:
                yield f"data: {json.dumps(snapshot)}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive comment so proxies don't drop the connection
                    yield ": keep-alive\n\n"
        finally:
            with _lock:
                if queue in _subscribers:
                    _subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables nginx buffering if you're behind one
        },
    )


# ── internal broadcast helper (called from MQTT thread) ──────────────────────

def _broadcast(payload: dict):
    """Push a new payload to every connected SSE client."""
    if _loop is None:
        return
    with _lock:
        for queue in _subscribers:
            _loop.call_soon_threadsafe(queue.put_nowait, payload)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    topics = load_yaml("config/topics.yaml")["topics"]

    broker = mqtt_cfg["broker"]
    client_cfg = mqtt_cfg["client"]

    baby_topic = topics["baby_state"]

    mqtt = MqttClient(
        client_id="api_service",
        host=broker["host"],
        port=broker["port"],
        keepalive=client_cfg.get("keepalive", 60),
        username=broker.get("username"),
        password=broker.get("password"),
    )
    mqtt.connect()

    def on_baby_state(topic: str, payload: dict):
        global _latest_state
        with _lock:
            _latest_state = payload
        _broadcast(payload)

    mqtt.subscribe(baby_topic, on_baby_state, qos=1)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
