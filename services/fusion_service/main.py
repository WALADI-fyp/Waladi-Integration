import time
import yaml

from shared.mqtt_client import MqttClient
from shared.message import make_message
from config.device import get_device_id

device_id = get_device_id()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    topics = load_yaml("config/topics.yaml")["topics"]

    sht_topic = topics["sht31_env"]
    vitals_topic = topics["vital_signs"]
    state_topic = topics["baby_state"]

    client = MqttClient(
        client_id=f"fusion_{device_id}",
        host=mqtt_cfg["broker"]["host"],
        port=mqtt_cfg["broker"]["port"],
        keepalive=mqtt_cfg["client"]["keepalive"],
        username=mqtt_cfg["broker"]["username"],
        password=mqtt_cfg["broker"]["password"],
        tls=mqtt_cfg["broker"].get("tls", False),
    )
    client.connect()

    latest_env = {
        "room_temp_c": None,
        "humidity_rh": None,
        "mock": True,
    }
    latest_vitals = {
        "breathing_rate_bpm": None,
        "heart_rate_bpm": None,
        "mock": True,
    }

    def env_callback(topic: str, msg: dict):
        nonlocal latest_env

        data = msg.get("data", {})
        latest_env = {
            "room_temp_c": data.get("room_temp_c"),
            "humidity_rh": data.get("humidity_rh"),
            "mock": data.get("mock", True),
        }
        print(f"[fusion] env update: {latest_env}")

    def vitals_callback(topic: str, msg: dict):
        nonlocal latest_vitals

        data = msg.get("data", {})
        latest_vitals = {
            "breathing_rate_bpm": data.get("breathing_rate_bpm"),
            "heart_rate_bpm": data.get("heart_rate_bpm"),
            "mock": data.get("mock", True),
        }
        print(f"[fusion] vitals update: {latest_vitals}")

    client.subscribe(sht_topic, env_callback, qos=1)
    client.subscribe(vitals_topic, vitals_callback, qos=1)

    counter = 0

    try:
        while True:
            room_temp = latest_env.get("room_temp_c")
            room_humidity = latest_env.get("humidity_rh")
            breathing_rate = latest_vitals.get("breathing_rate_bpm")
            heart_rate = latest_vitals.get("heart_rate_bpm")

            field_validity = {
                "breathing_rate_bpm": breathing_rate is not None and not latest_vitals.get("mock", True),
                "heart_rate_bpm": heart_rate is not None and not latest_vitals.get("mock", True),
                "room_temperature_c": room_temp is not None and not latest_env.get("mock", True),
                "body_temperature_c": False,
                "room_humidity_rh": room_humidity is not None and not latest_env.get("mock", True),
            }

            state = make_message(
                source="fusion_service",
                data={
                    "breathing_rate_bpm": breathing_rate if breathing_rate is not None else counter,
                    "heart_rate_bpm": heart_rate if heart_rate is not None else counter,
                    "room_temperature_c": room_temp if room_temp is not None else counter,
                    "body_temperature_c": counter,  # still dummy for now
                    "room_humidity_rh": room_humidity if room_humidity is not None else counter,
                    "mock_fields": [k for k, is_real in field_validity.items() if not is_real],
                    "device_id": device_id,
                },
            )

            client.publish_json(state_topic, state, qos=1, retain=True)
            print(f"[fusion] published -> {state_topic}: {state}")

            counter += 1
            time.sleep(1.0)
    finally:
        client.close()


if __name__ == "__main__":
    main()
