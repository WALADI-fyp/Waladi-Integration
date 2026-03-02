import json
import ssl
from typing import Callable, Optional

import paho.mqtt.client as mqtt


class MqttClient:
    
    #- connect() starts a background loop so messages flow.
    #- publish_json() sends Python dicts as JSON strings.
    #- subscribe() receives messages and parses JSON back to dict.
    

    def __init__(
        self,
        *,
        client_id: str,
        host: str,
        port: int,
        keepalive: int = 30,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tls: bool = False,
    ):
        self.host = host
        self.port = port
        self.keepalive = keepalive

        self.client = mqtt.Client(client_id=client_id, clean_session=True)

        if username:
            self.client.username_pw_set(username, password=password)

        if tls:
            # EMQX Cloud serverless uses a CA-signed cert — system roots are fine.
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        codes = {
            0: "OK",
            1: "bad protocol",
            2: "client ID rejected",
            3: "server unavailable",
            4: "bad credentials",
            5: "not authorised",
        }
        print(f"[MQTT] connected rc={rc} ({codes.get(rc, 'unknown')})")

    def _on_disconnect(self, client, userdata, rc):
        print(f"[MQTT] disconnected rc={rc}")

    def connect(self):
        self.client.connect(self.host, self.port, keepalive=self.keepalive)
        self.client.loop_start()

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()

    def publish_json(self, topic: str, payload: dict, qos: int = 1, retain: bool = False):
        data = json.dumps(payload, separators=(",", ":"))
        self.client.publish(topic, data, qos=qos, retain=retain)

    def subscribe(self, topic: str, on_message: Callable[[str, dict], None], qos: int = 1):

        def _handler(client, userdata, msg):
            try:
                obj = json.loads(msg.payload.decode("utf-8"))
            except Exception as e:
                print(f"[MQTT] bad json on {msg.topic}: {e}")
                return
            on_message(msg.topic, obj)

        self.client.message_callback_add(topic, _handler)
        self.client.subscribe(topic, qos=qos)
