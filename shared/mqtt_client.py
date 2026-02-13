import json
from typing import Callable, Optional

import paho.mqtt.client as mqtt


class MqttClient:
    """
    Small wrapper around paho-mqtt.
    - connect() starts a background loop so messages flow.
    - publish_json() sends Python dicts as JSON strings.
    - subscribe() receives messages and parses JSON back to dict.
    """

    def __init__(
        self,
        *,
        client_id: str,
        host: str,
        port: int,
        keepalive: int = 30,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.keepalive = keepalive

        # clean_session=True means: don't resume old subscriptions after reconnect.
        # For our first version, this keeps behavior simple and predictable.
        self.client = mqtt.Client(client_id=client_id, clean_session=True)

        if username:
            self.client.username_pw_set(username, password=password)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        # rc == 0 means success
        print(f"[MQTT] connected rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        print(f"[MQTT] disconnected rc={rc}")

    def connect(self):
        """
        Connect to the broker and start the network loop in a background thread.
        """
        self.client.connect(self.host, self.port, keepalive=self.keepalive)
        self.client.loop_start()

    def close(self):
        """
        Stop background loop and disconnect cleanly.
        """
        self.client.loop_stop()
        self.client.disconnect()

    def publish_json(self, topic: str, payload: dict, qos: int = 1, retain: bool = False):
        """
        Publish a Python dict as JSON to a topic.
        qos=1 is a good default for sensor data and state.
        """
        data = json.dumps(payload, separators=(",", ":"))
        self.client.publish(topic, data, qos=qos, retain=retain)

    def subscribe(self, topic: str, on_message: Callable[[str, dict], None], qos: int = 1):
        """
        Subscribe to a topic and call on_message(topic, dict_payload) for each message.
        """

        def _handler(client, userdata, msg):
            try:
                obj = json.loads(msg.payload.decode("utf-8"))
            except Exception as e:
                print(f"[MQTT] bad json on {msg.topic}: {e}")
                return
            on_message(msg.topic, obj)

        # message_callback_add attaches a handler specifically for this topic.
        self.client.message_callback_add(topic, _handler)
        self.client.subscribe(topic, qos=qos)
