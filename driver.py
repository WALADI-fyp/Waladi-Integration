import json
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
import subprocess

import yaml

from config.device import get_device_id
from shared.message import make_message
from shared.mqtt_client import MqttClient

SERVICES = [
    "services.sht31_service.main",
    "services.fusion_service.main",
    "services.camera_service.main",
    "services.db_writer_service.main",
    "services.mmwave_vitals_service.main",
    "services.thermal_camera_service.main",
    "services.ai_pose_service.main",
    "services.cry_detection_service.main",
    "services.temperature_alert_service.main",
    "services.vital_alert_service.main",
    "services.sleep_detection_service.main",
]

processes = []
_original_terminal_settings = None
_demo_thread = None
_demo_lock = threading.Lock()
_demo_stop_event = threading.Event()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def restore_terminal():
    global _original_terminal_settings
    if _original_terminal_settings is not None:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _original_terminal_settings)
        except Exception:
            pass
        _original_terminal_settings = None


def setup_single_key_input():
    """Allow pressing 'a' without Enter on Raspberry Pi/Linux terminals."""
    global _original_terminal_settings
    try:
        if sys.stdin.isatty():
            _original_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
    except Exception as e:
        print(f"[driver] keyboard hotkey disabled: {e}")


def shutdown(sig=None, frame=None):
    print("\n[driver] shutting down...")
    _demo_stop_event.set()
    restore_terminal()
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    sys.exit(0)


def make_demo_mqtt_client() -> tuple[MqttClient, str, str]:
    mqtt_cfg = load_yaml("config/mqtt.yaml")
    topics = load_yaml("config/topics.yaml")["topics"]
    device_id = get_device_id()

    broker_cfg = mqtt_cfg["broker"]
    client_cfg = mqtt_cfg["client"]
    client = MqttClient(
        client_id=f"driver_demo_breath_{device_id}",
        host=broker_cfg["host"],
        port=broker_cfg["port"],
        keepalive=client_cfg.get("keepalive", 60),
        username=broker_cfg.get("username"),
        password=broker_cfg.get("password"),
        tls=broker_cfg.get("tls", False),
    )
    client.connect()

    # Use a dedicated demo topic so the fake breath-rate countdown does not
    # overwrite the real mmWave heart_rate_bpm in fusion/db services.
    demo_topic = topics.get("breath_rate_demo", "waladi/demo/breath_rate")
    return client, demo_topic, device_id


def breath_rate_demo_worker():
    """
    Publishes fake breath-rate-only readings to a dedicated demo topic.

    Starts breath rate at 21 and decreases by 1 every second down to 8.
    It intentionally does NOT publish heart_rate_bpm, so the real heartbeat
    values from the mmWave service are not touched or overwritten.
    """
    client = None
    try:
        client, vitals_topic, device_id = make_demo_mqtt_client()
        print("[driver-demo] breath-rate demo started: 21 -> 8, decreasing by 1 every second")

        for breath_rate in range(21, 7, -1):
            if _demo_stop_event.is_set():
                break

            data = {
                "breathing_rate_bpm": float(breath_rate),
                "mock": False,
                "fake_demo": True,
                "device_id": device_id,
            }
            client.publish_json(
                vitals_topic,
                make_message(source="driver_breath_rate_demo", data=data),
                qos=1,
                retain=False,
            )
            print(f"[driver-demo] published fake breathing_rate_bpm={breath_rate}; heart_rate_bpm unchanged")
            time.sleep(1.0)

        print("[driver-demo] breath-rate demo finished")
    except Exception as e:
        print(f"[driver-demo] ERROR: {e}")
    finally:
        try:
            if client is not None:
                client.close()
        except Exception:
            pass


def start_breath_rate_demo():
    global _demo_thread
    with _demo_lock:
        if _demo_thread is not None and _demo_thread.is_alive():
            print("[driver-demo] demo is already running")
            return
        _demo_stop_event.clear()
        _demo_thread = threading.Thread(target=breath_rate_demo_worker, daemon=True)
        _demo_thread.start()


def handle_keyboard_hotkeys():
    """Press 'a' to start the fake breath-rate countdown."""
    try:
        if not sys.stdin.isatty():
            return
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return
        key = sys.stdin.read(1)
        if key.lower() == "a":
            start_breath_rate_demo()
        elif key.lower() == "q":
            shutdown(None, None)
    except Exception as e:
        print(f"[driver] keyboard read error: {e}")


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


if __name__ == "__main__":
    print("[driver] starting Waladi backend...")

    setup_single_key_input()

    for service in SERVICES:
        p = subprocess.Popen(
            [sys.executable, "-m", service],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        processes.append(p)
        print(f"[driver] started {service} (pid={p.pid})")

    print("\n[driver] all services running — only errors and key events will print")
    print("[driver] hotkeys: press 'a' to fake breath rate 21 -> 8, press 'q' to quit\n")

    while True:
        handle_keyboard_hotkeys()
        for p in processes:
            if p.poll() is not None:
                idx = processes.index(p)
                name = SERVICES[idx] if idx < len(SERVICES) else "unknown"
                print(f"[driver] ERROR: {name} exited unexpectedly (pid={p.pid}), shutting down")
                shutdown(None, None)
        time.sleep(0.1)
