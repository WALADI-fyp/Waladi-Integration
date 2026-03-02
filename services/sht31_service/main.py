import time
import yaml

from shared.mqtt_client import MqttClient
from shared.message import make_message
from services.sht31_service.sht31_driver import SHT31
from config.device import get_device_id
device_id = get_device_id()

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
 

    username = mqtt_cfg["broker"].get("username")
    password = mqtt_cfg["broker"].get("password")

    topic = topics_cfg["sht31_env"]

    # 2) Connect to MQTT broker
    tls = mqtt_cfg["broker"].get("tls", False)

    client = MqttClient(
        client_id=f"sht31_{device_id}",
        host=host,
        port=port,
        keepalive=keepalive,
        username=username,
        password=password,
        tls=tls,
    )
    client.connect()

    # 3) Init sensor 
    sensor = SHT31(bus_id=1, address=0x44)

    counter = 0

    try:
        while True:
            try:
                temp_c, humidity_rh = sensor.read()
                data = {
                    "room_temp_c": round(float(temp_c), 2),
                    "humidity_rh": round(float(humidity_rh), 2),
                    "mock": False,
                }
            except Exception as e:
                # Sensor not connected or read failed — publish incrementing mock values
                print(f"[sht31_service] sensor unavailable ({e}), using mock counter={counter}")
                data = {
                    "room_temp_c": counter,
                    "humidity_rh": counter,
                    "mock": True,
                }
                counter += 1

            msg = make_message(source="sht31_service", data=data)

            client.publish_json(topic, msg, qos=1, retain=False)
            print(f"published -> {topic}: {msg}")

            time.sleep(1.0)
    finally:
        client.close()


if __name__ == "__main__":
    main()
