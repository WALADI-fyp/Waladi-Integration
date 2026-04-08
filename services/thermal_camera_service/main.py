import time
import yaml

from services.thermal_camera_service.mlx90640_driver import MLX90640Driver
from shared.message import make_message
from shared.mqtt_client import MqttClient


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Load configs
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    topics_root = load_yaml("config/topics.yaml")

    broker_cfg = mqtt_cfg["broker"]
    client_cfg = mqtt_cfg["client"]
    topics_cfg = topics_root["topics"]

    host = broker_cfg["host"]
    port = broker_cfg["port"]
    username = broker_cfg.get("username")
    password = broker_cfg.get("password")
    tls = broker_cfg.get("tls", False)

    keepalive = client_cfg.get("keepalive", 60)
    client_id = client_cfg["device_id"]

    topic = topics_cfg["thermal_hotspot"]

    # Init MQTT
    client = MqttClient(
        host=host,
        port=port,
        client_id=client_id,
        keepalive=keepalive,
        username=username,
        password=password,
        tls=tls,
    )
    client.connect()

    # Init thermal camera
    thermal = MLX90640Driver(enable_visualization=True)

    try:
        while True:
            try:
                result = thermal.read()

                data = {
                    "max_temp_c": round(result["max_temp_c"], 2),
                    "max_row": result["max_row"],
                    "max_col": result["max_col"],
                    "min_temp_c": round(result["min_temp_c"], 2),
                    "avg_temp_c": round(result["avg_temp_c"], 2),
                    "x_norm": round(result["max_col"] / (result["width"] - 1), 4),
                    "y_norm": round(result["max_row"] / (result["height"] - 1), 4),
                    "height": result["height"],
                    "width": result["width"],
                    "mock": False,
                }

            except Exception as e:
                print(f"[thermal_camera_service] Thermal read failed: {e}")
                data = {
                    "max_temp_c": None,
                    "max_row": None,
                    "max_col": None,
                    "min_temp_c": None,
                    "avg_temp_c": None,
                    "x_norm": None,
                    "y_norm": None,
                    "height": 24,
                    "width": 32,
                    "mock": True,
                    "error": str(e),
                }

            msg = make_message(source="thermal_camera_service", data=data)
            client.publish_json(topic, msg, qos=1, retain=False)
            print(f"published -> {topic}: {msg}")

            time.sleep(0.25)

    finally:
        thermal.close()
        client.close()


if __name__ == "__main__":
    main()
