import time
import yaml

from shared.mqtt_client import MqttClient
from shared.message import make_message


def load_yaml(path: str) -> dict:
    """Load YAML into a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # --------------------------
    # 1) Load configuration
    # --------------------------
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    topics = load_yaml("config/topics.yaml")["topics"]

    device_id = mqtt_cfg["client"]["device_id"]

    sht_topic = topics["sht31_env"]
    state_topic = topics["baby_state"]

    # --------------------------
    # 2) Connect to broker
    # --------------------------
    client = MqttClient(
        client_id="fusion_service",
        host=mqtt_cfg["broker"]["host"],
        port=mqtt_cfg["broker"]["port"],
        keepalive=mqtt_cfg["client"]["keepalive"],
        username=mqtt_cfg["broker"]["username"],
        password=mqtt_cfg["broker"]["password"],
    )
    client.connect()

    # --------------------------
    # 3) In-memory "latest values"
    # --------------------------
    latest_env = {"temp_c": None, "humidity_rh": None}
    latest_env_ts = None  # timestamp of last env message received

    # --------------------------
    # 4) Subscriber callback
    # --------------------------
    def on_env(topic: str, msg: dict):
        nonlocal latest_env, latest_env_ts

        # msg is the standard structure: {ts, device_id, source, data}
        data = msg.get("data", {})

        latest_env = {
            "temp_c": data.get("temp_c"),
            "humidity_rh": data.get("humidity_rh"),
        }
        latest_env_ts = msg.get("ts")

        print(f"[fusion] env update: {latest_env}")

    client.subscribe(sht_topic, on_env, qos=1)

    # --------------------------
    # 5) Publish combined state
    # --------------------------
    try:
        while True:
            # Simple "freshness" indicator: do we have recent env?
            now = int(time.time() * 1000)
            env_age_ms = None if latest_env_ts is None else (now - latest_env_ts)

            state = make_message(
                device_id=device_id,
                source="fusion_service",
                data={
                    "env": latest_env,
                    "env_age_ms": env_age_ms,
                    "breathing": {"status": "unknown"},
                    "audio": {"status": "unknown"},
                },
            )

            # retain=True so any new subscriber instantly gets latest state
            client.publish_json(state_topic, state, qos=1, retain=True)
            time.sleep(1.0)
    finally:
        client.close()


if __name__ == "__main__":
    main()
