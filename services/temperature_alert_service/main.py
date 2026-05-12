"""
temperature_alert_service
─────────────────────────
Listens to thermal hotspot readings and creates event-based temperature alerts.

No start/end lifecycle is used. The service inserts and publishes one alert when
crossing into a new severity band, then suppresses repeats until the severity
changes or temperature returns to normal.

Run from project root:
  python3 -m services.temperature_alert_service.main
"""

import time
from datetime import datetime, timezone
from typing import Optional

import yaml

from config.device import get_device_id
from shared.db_client import DbClient
from shared.message import make_message, now_ms
from shared.mqtt_client import MqttClient


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def classify_temperature(temp_c: float) -> Optional[str]:
    """Return alert severity, or None when no alert should be emitted."""
    if temp_c <= 37.0:
        return None
    if temp_c <= 37.5:
        return "normal_high"
    if temp_c <= 38.0:
        return "moderately_high"
    return "severe"


def extract_temperature(payload: dict) -> Optional[float]:
    """Support both wrapped make_message payloads and direct payloads."""
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    for key in ("max_temp_c", "temperature_c", "temp_c", "value"):
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            print(f"[temp] invalid temperature field {key}={value!r}")
            return None
    print("[temp] missing temperature field in thermal payload")
    return None


def resolve_user_id(db: DbClient, device_id: str) -> str:
    user_id = None
    for attempt in range(3):
        try:
            user_id = db.get_user_id(device_id)
        except Exception as e:
            print(f"[temp] DB error looking up user_id: {e}")
        if user_id:
            print(f"[temp] paired to user_id={user_id}")
            return user_id
        print(f"[temp] device not yet paired (attempt {attempt + 1}/3) — retrying in 5s")
        time.sleep(5)

    print("[temp] no pairing found — writing alerts with user_id='unassigned'")
    return "unassigned"


def main():
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    db_cfg = load_yaml("config/db.yaml")["timescale"]
    topics = load_yaml("config/topics.yaml")["topics"]

    device_id = get_device_id()
    thermal_topic = topics["thermal_hotspot"]
    alert_topic = topics["temperature_alert"]

    db = DbClient(
        host=db_cfg["host"],
        port=db_cfg["port"],
        dbname=db_cfg["dbname"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        sslmode=db_cfg.get("sslmode", "require"),
    )
    db.connect()
    db.init_temperature_alerts_table()
    user_id = resolve_user_id(db, device_id)

    broker_cfg = mqtt_cfg["broker"]
    client_cfg = mqtt_cfg["client"]
    mqtt = MqttClient(
        client_id=f"temperature_alert_{device_id}",
        host=broker_cfg["host"],
        port=broker_cfg["port"],
        keepalive=client_cfg.get("keepalive", 60),
        username=broker_cfg.get("username"),
        password=broker_cfg.get("password"),
        tls=broker_cfg.get("tls", False),
    )

    last_severity_by_device: dict[str, Optional[str]] = {}

    def on_thermal_message(topic: str, payload: dict):
        temp_c = extract_temperature(payload)
        if temp_c is None:
            return

        severity = classify_temperature(temp_c)
        msg_device_id = payload.get("data", {}).get("device_id") or payload.get("device_id") or device_id
        previous = last_severity_by_device.get(msg_device_id)

        if severity is None:
            if previous is not None:
                print(f"[temp] {temp_c:.2f}C back to normal; reset severity for {msg_device_id}")
            last_severity_by_device[msg_device_id] = None
            return

        if severity == previous:
            return

        ts = now_ms()
        created_at = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat()
        temperature_c = round(temp_c, 2)

        try:
            alert_id = db.insert_temperature_alert(
                user_id=user_id,
                device_id=msg_device_id,
                created_at_ms=ts,
                temperature_c=temperature_c,
                severity=severity,
            )
            print(f"[temp] alert #{alert_id} {severity} temperature={temperature_c}C")
        except Exception as e:
            print(f"[temp] DB insert failed: {e}")
            alert_id = None

        mqtt.publish_json(alert_topic, make_message(
            source="temperature_alert_service",
            data={
                "event": "temperature_alert",
                "user_id": user_id,
                "device_id": msg_device_id,
                "temperature_c": temperature_c,
                "severity": severity,
                "created_at": created_at,
                "alert_id": alert_id,
            },
        ))

        last_severity_by_device[msg_device_id] = severity

    mqtt.connect()
    mqtt.subscribe(thermal_topic, on_thermal_message, qos=1)
    print(f"[temp] service ready — listening to '{thermal_topic}', publishing to '{alert_topic}'")

    try:
        while True:
            time.sleep(1)
    finally:
        mqtt.close()
        db.close()


if __name__ == "__main__":
    main()
