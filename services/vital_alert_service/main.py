"""
vital_alert_service
───────────────────
Listens to mmWave vital readings and creates event-based alerts for:
- heart_rate
- breath_rate

No start/end lifecycle is used. One alert is inserted and published when a
vital crosses into a critical state. Repeats are suppressed until that same
vital returns to non-critical or changes critical direction.

Run from project root:
  python3 -m services.vital_alert_service.main
"""

import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import yaml

from config.device import get_device_id
from shared.db_client import DbClient
from shared.message import make_message, now_ms
from shared.mqtt_client import MqttClient


HEART_RATE = "heart_rate"
BREATH_RATE = "breath_rate"


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def safe_float(value, field_name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"[vital] invalid {field_name}={value!r}")
        return None


def classify_heart_rate(heart_rate_bpm: float) -> Optional[str]:
    """Return alert severity, or None when heart rate is not critical."""
    if heart_rate_bpm < 80.0:
        return "critical_low"
    if heart_rate_bpm > 200.0:
        return "critical_high"
    return None


def classify_breath_rate(breathing_rate_bpm: float) -> Optional[str]:
    """Return alert severity, or None when breath rate is not critical."""
    if breathing_rate_bpm < 20.0:
        return "critical_low"
    if breathing_rate_bpm > 60.0:
        return "critical_high"
    return None


def message_for(vital_type: str, severity: str) -> str:
    if vital_type == HEART_RATE and severity == "critical_low":
        return "Heart rate is critically low"
    if vital_type == HEART_RATE and severity == "critical_high":
        return "Heart rate is critically high"
    if vital_type == BREATH_RATE and severity == "critical_low":
        return "Breath rate is critically low"
    if vital_type == BREATH_RATE and severity == "critical_high":
        return "Breath rate is critically high"
    return "Vital sign is critical"


def unit_for(vital_type: str) -> str:
    if vital_type == HEART_RATE:
        return "bpm"
    if vital_type == BREATH_RATE:
        return "breaths_per_min"
    return ""


def extract_vitals(payload: dict) -> Tuple[Optional[float], Optional[float], bool, Optional[str]]:
    """
    Supports wrapped make_message payloads and direct payloads.
    Returns: breathing_rate_bpm, heart_rate_bpm, is_mock, payload_device_id.
    """
    data = payload.get("data", payload) if isinstance(payload, dict) else {}

    breathing_rate = safe_float(
        data.get("breathing_rate_bpm", data.get("breath_rate_bpm", data.get("breath_rate"))),
        "breathing_rate_bpm",
    )
    heart_rate = safe_float(
        data.get("heart_rate_bpm", data.get("heart_rate")),
        "heart_rate_bpm",
    )
    is_mock = bool(data.get("mock", False))
    payload_device_id = data.get("device_id") or payload.get("device_id") if isinstance(payload, dict) else None
    return breathing_rate, heart_rate, is_mock, payload_device_id


def resolve_user_id(db: DbClient, device_id: str) -> str:
    user_id = None
    for attempt in range(3):
        try:
            user_id = db.get_user_id(device_id)
        except Exception as e:
            print(f"[vital] DB error looking up user_id: {e}")
        if user_id:
            print(f"[vital] paired to user_id={user_id}")
            return user_id
        print(f"[vital] device not yet paired (attempt {attempt + 1}/3) — retrying in 5s")
        time.sleep(5)

    print("[vital] no pairing found — writing alerts with user_id='unassigned'")
    return "unassigned"


def main():
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    db_cfg = load_yaml("config/db.yaml")["timescale"]
    topics = load_yaml("config/topics.yaml")["topics"]

    device_id = get_device_id()
    vitals_topic = topics["vital_signs"]
    alert_topic = topics["vital_alert"]

    db = DbClient(
        host=db_cfg["host"],
        port=db_cfg["port"],
        dbname=db_cfg["dbname"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        sslmode=db_cfg.get("sslmode", "require"),
    )
    db.connect()
    db.init_vital_alerts_table()
    user_id = resolve_user_id(db, device_id)

    broker_cfg = mqtt_cfg["broker"]
    client_cfg = mqtt_cfg["client"]
    mqtt = MqttClient(
        client_id=f"vital_alert_{device_id}",
        host=broker_cfg["host"],
        port=broker_cfg["port"],
        keepalive=client_cfg.get("keepalive", 60),
        username=broker_cfg.get("username"),
        password=broker_cfg.get("password"),
        tls=broker_cfg.get("tls", False),
    )

    # Dedup is per device + vital_type.
    last_severity_by_key: dict[tuple[str, str], Optional[str]] = {}

    def maybe_emit_alert(*, msg_device_id: str, vital_type: str, value: float, severity: Optional[str]):
        key = (msg_device_id, vital_type)
        previous = last_severity_by_key.get(key)

        if severity is None:
            if previous is not None:
                print(f"[vital] {vital_type}={value:.1f} back to non-critical; reset for {msg_device_id}")
            last_severity_by_key[key] = None
            return

        if severity == previous:
            return

        ts = now_ms()
        created_at = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat()
        rounded_value = round(value, 2)
        message = message_for(vital_type, severity)
        unit = unit_for(vital_type)

        try:
            alert_id = db.insert_vital_alert(
                user_id=user_id,
                device_id=msg_device_id,
                created_at_ms=ts,
                vital_type=vital_type,
                value=rounded_value,
                severity=severity,
                message=message,
            )
            print(f"[vital] alert #{alert_id} {vital_type} {severity} value={rounded_value}")
        except Exception as e:
            print(f"[vital] DB insert failed: {e}")
            alert_id = None

        mqtt.publish_json(alert_topic, make_message(
            source="vital_alert_service",
            data={
                "event": "vital_alert",
                "user_id": user_id,
                "device_id": msg_device_id,
                "vital_type": vital_type,
                "value": rounded_value,
                "unit": unit,
                "severity": severity,
                "message": message,
                "created_at": created_at,
                "alert_id": alert_id,
            },
        ))

        last_severity_by_key[key] = severity

    def on_vitals_message(topic: str, payload: dict):
        breathing_rate, heart_rate, is_mock, payload_device_id = extract_vitals(payload)
        msg_device_id = payload_device_id or device_id

        # The mmWave service publishes counter values when the sensor is unavailable.
        # Do not create alerts from those mock values.
        if is_mock:
            return

        if heart_rate is not None:
            maybe_emit_alert(
                msg_device_id=msg_device_id,
                vital_type=HEART_RATE,
                value=heart_rate,
                severity=classify_heart_rate(heart_rate),
            )

        if breathing_rate is not None:
            maybe_emit_alert(
                msg_device_id=msg_device_id,
                vital_type=BREATH_RATE,
                value=breathing_rate,
                severity=classify_breath_rate(breathing_rate),
            )

        if heart_rate is None and breathing_rate is None:
            print("[vital] missing heart_rate_bpm and breathing_rate_bpm in vitals payload")

    mqtt.connect()
    mqtt.subscribe(vitals_topic, on_vitals_message, qos=1)
    print(f"[vital] service ready — listening to '{vitals_topic}', publishing to '{alert_topic}'")

    try:
        while True:
            time.sleep(1)
    finally:
        mqtt.close()
        db.close()


if __name__ == "__main__":
    main()
