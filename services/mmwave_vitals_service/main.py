import time
import yaml

from shared.mqtt_client import MqttClient
from shared.message import make_message
from config.device import get_device_id
from .mmwave_driver import MmwaveVitalsSensor

device_id = get_device_id()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    topics   = load_yaml("config/topics.yaml")["topics"]

    vitals_topic = topics["vital_signs"]

    client = MqttClient(
        client_id=f"mmwave_vitals_{device_id}",
        host=mqtt_cfg["broker"]["host"],
        port=mqtt_cfg["broker"]["port"],
        keepalive=mqtt_cfg["client"]["keepalive"],
        username=mqtt_cfg["broker"]["username"],
        password=mqtt_cfg["broker"]["password"],
        tls=mqtt_cfg["broker"].get("tls", False),
    )
    client.connect()

    sensor = MmwaveVitalsSensor(
        port="/dev/ttyACM0",
        baudrate=115200,
        timeout=1.0,
    )

    counter = 0

    try:
        sensor.connect()
        print("[mmwave_vitals_service] sensor connected, reading vitals...")

        while True:
            try:
                breathing_rate_bpm, heart_rate_bpm = sensor.read(max_wait_s=5.0)

                is_mock = (breathing_rate_bpm is None or heart_rate_bpm is None)

                data = {
                    "breathing_rate_bpm": breathing_rate_bpm if breathing_rate_bpm is not None else counter,
                    "heart_rate_bpm":     heart_rate_bpm     if heart_rate_bpm     is not None else counter,
                    "mock": is_mock,
                }

            except Exception as e:
                print(f"[mmwave_vitals_service] sensor read failed: {e}")
                data = {
                    "breathing_rate_bpm": counter,
                    "heart_rate_bpm":     counter,
                    "mock": True,
                    "error": str(e),
                }
                counter += 1

            msg = make_message(source="mmwave_vitals_service", data=data)
            client.publish_json(vitals_topic, msg, qos=1, retain=False)
            print(f"[mmwave_vitals_service] published -> {vitals_topic}: {msg}")

            time.sleep(1.0)

    finally:
        try:
            sensor.close()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
