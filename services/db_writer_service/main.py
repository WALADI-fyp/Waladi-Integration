import time
import yaml

from config.device import get_device_id
from shared.mqtt_client import MqttClient
from shared.db_client import DbClient


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # ── Load configs ────────────────────────────────────────────────────────
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    topics   = load_yaml("config/topics.yaml")["topics"]
    db_cfg   = load_yaml("config/db.yaml")["timescale"]

    device_id   = get_device_id()
    state_topic = topics["baby_state"]

    # ── Connect to TimescaleDB ───────────────────────────────────────────────
    db = DbClient(
        host=db_cfg["host"],
        port=db_cfg["port"],
        dbname=db_cfg["dbname"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        sslmode=db_cfg.get("sslmode", "require"),
    )
    db.connect()
    db.init_db()

    # ── Resolve user_id (retry until device is paired via the app) ───────────
    user_id = None
    print(f"[db_writer] waiting for device '{device_id}' to be paired in app...")
    while user_id is None:
        try:
            user_id = db.get_user_id(device_id)
        except Exception as e:
            print(f"[db_writer] DB error looking up user_id: {e}")
        if user_id is None:
            print("[db_writer] device not yet paired — retrying in 10s")
            time.sleep(10)

    print(f"[db_writer] paired to user_id={user_id}")

    # ── Connect to EMQX and subscribe to fused state ─────────────────────────
    mqtt = MqttClient(
        client_id=f"db_writer_{device_id}",
        host=mqtt_cfg["broker"]["host"],
        port=mqtt_cfg["broker"]["port"],
        keepalive=mqtt_cfg["client"]["keepalive"],
        username=mqtt_cfg["broker"].get("username"),
        password=mqtt_cfg["broker"].get("password"),
        tls=mqtt_cfg["broker"].get("tls", False),
    )
    mqtt.connect()

    def on_state(topic: str, msg: dict):
        """Called for every message on state/baby — write it straight to TimescaleDB."""
        try:
            ts_ms  = msg.get("ts", int(time.time() * 1000))
            source = msg.get("source", "fusion_service")
            data   = msg.get("data", {})

            db.insert_reading(
                user_id=user_id,
                device_id=device_id,
                ts_ms=ts_ms,
                room_temperature_c=data.get("room_temperature_c"),
                room_humidity_rh=data.get("room_humidity_rh"),
                breathing_rate_bpm=data.get("breathing_rate_bpm"),
                heart_rate_bpm=data.get("heart_rate_bpm"),
                body_temperature_c=data.get("body_temperature_c"),
                mock_fields=data.get("mock_fields", []),
                source=source,
            )
            print(f"[db_writer] ✓ wrote reading to TimescaleDB (ts={ts_ms})")

        except Exception as e:
            print(f"[db_writer] ✗ insert failed: {e}")
            # Force reconnect on next insert
            try:
                db.close()
            except Exception:
                pass

    mqtt.subscribe(state_topic, on_state, qos=1)
    print(f"[db_writer] subscribed to '{state_topic}' — writing all readings to TimescaleDB")

    try:
        while True:
            time.sleep(1)
    finally:
        mqtt.close()
        db.close()


if __name__ == "__main__":
    main()
