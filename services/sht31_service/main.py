import time
import yaml

from shared.mqtt_client import MqttClient
from shared.message import make_message
from services.sht31_service.sht31_driver import SHT31


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1) Load configs
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    topics_cfg = load_yaml("config/topics.yaml")["topics"]

    host = mqtt_cfg["broker"]["host"]
    port = mqtt_cfg["broker"]["port"]
    keepalive = mqtt_cfg["client"]["keepalive"]
    device_id = mqtt_cfg["client"]["device_id"]

    username = mqtt_cfg["broker"].get("username")
    password = mqtt_cfg["broker"].get("password")

    topic = topics_cfg["sht31_env"]

    # 2) Connect to MQTT broker
    client = MqttClient(
        client_id="sht31_service",
        host=host,
        port=port,
        keepalive=keepalive,
        username=username,
        password=password,
    )
    client.connect()

    # 3) Initsensor 
    sensor = SHT31(bus_id=1, address=0x44)

    try:
        while True:
            try:
                temp_c, humidity_rh = sensor.read()
            except Exception as e:
                print(f"[sht31_service] read failed: {e}")
                time.sleep(1.0)
                continue

            msg = make_message(
                device_id=device_id,
                source="sht31_service",
                data={
                    "temp_c": round(float(temp_c), 2),
                    "humidity_rh": round(float(humidity_rh), 2),
                },
            )

            client.publish_json(topic, msg, qos=1, retain=False)
            print(f"published -> {topic}: {msg}")

            time.sleep(1.0)
    finally:
        client.close()


if __name__ == "__main__":
    main()
