import time
import yaml

from shared.mqtt_client import MqttClient
from shared.message import make_message


def load_yaml(path: str) -> dict:
    """Load a YAML file into a Python dict."""
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

    username = mqtt_cfg["broker"]["username"]
    password = mqtt_cfg["broker"]["password"]

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

    # 3) Dummy readings (we will replace this later with real SHT31 reads)
    temp_c = 24.0
    humidity_rh = 50.0

    try:
        while True:
            # Change values slightly so you can see updates
            temp_c += 0.05
            humidity_rh += 0.10

            msg = make_message(
                device_id=device_id,
                source="sht31_service",
                data={
                    "temp_c": round(temp_c, 2),
                    "humidity_rh": round(humidity_rh, 2),
                },
            )

            # Publish to MQTT
            client.publish_json(topic, msg, qos=1, retain=False)
            print(f"published -> {topic}: {msg}")

            time.sleep(1.0)
    finally:
        client.close()


if __name__ == "__main__":
    main()
