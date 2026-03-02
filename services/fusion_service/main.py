import time
import yaml

from shared.mqtt_client import MqttClient
from shared.message import make_message

from config.device import get_device_id
device_id = get_device_id()

def load_yaml(path: str) -> dict:
    """Load YAML into a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
  
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    topics = load_yaml("config/topics.yaml")["topics"]

    sht_topic = topics["sht31_env"]
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

    

    latest_env = {"temp_c": None, "humidity_rh": None}
    latest_env_ts = None  

 
    def env_callback(topic: str, msg: dict):
        nonlocal latest_env, latest_env_ts

     
        data = msg.get("data", {})

        latest_env = {
            "room_temp_c": data.get("room_temp_c"),
            "humidity_rh": data.get("humidity_rh"),
        }
        latest_env_ts = msg.get("ts")

        print(f"[fusion] env update: {latest_env}")

    client.subscribe(sht_topic, env_callback, qos=1)


    counter = 0

    try:
        while True:
            room_temp    = latest_env.get("room_temp_c")
            room_humidity = latest_env.get("humidity_rh")

            state = make_message(
                source="fusion_service",
                data={
                    # Real sensor value if available, otherwise incrementing mock
                    "breathing_rate_bpm":  counter if True else None,  # no sensor yet
                    "heart_rate_bpm":      counter if True else None,  # no sensor yet
                    "room_temperature_c":  room_temp  if room_temp  is not None else counter,
                    "body_temperature_c":  counter,                    # no sensor yet
                    "room_humidity_rh":    room_humidity if room_humidity is not None else counter,
                    # Tells your app which values are real vs mock placeholders
                    "mock_fields": [
                        f for f, v in {
                            "breathing_rate_bpm": False,
                            "heart_rate_bpm":     False,
                            "room_temperature_c": room_temp is not None,
                            "body_temperature_c": False,
                            "room_humidity_rh":   room_humidity is not None,
                        }.items() if not v
                    ],
                },
            )

            client.publish_json(state_topic, state, qos=1, retain=True)
            counter += 1
            time.sleep(1.0)
    finally:
        client.close()


if __name__ == "__main__":
    main()
