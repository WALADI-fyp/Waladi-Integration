import threading
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI
import uvicorn

from shared.mqtt_client import MqttClient

app = FastAPI()

_latest_state: Optional[Dict[str, Any]] = None
_lock = threading.Lock()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@app.get("/status")
def health():
    return {"status": "ok"}


@app.get("/state")
def state():
    with _lock:
        return _latest_state if _latest_state is not None else {"status": "no_state_yet"}


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

    mqtt.subscribe(baby_topic, on_baby_state, qos=1)

    # HTTP server 
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
