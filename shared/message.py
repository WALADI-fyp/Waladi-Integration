import time
from typing import Any, Dict


def now_ms() -> int:
    """
    Returns the current time in milliseconds since Unix epoch.
    We use ms so we can compare timestamps easily across sensors.
    """
    return int(time.time() * 1000)


def make_message(*, device_id: str, source: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wraps a payload in a standard structure used by all MQTT topics.

    device_id: identifies which device (Pi) produced the message
    source: identifies which service produced the message
    data: the actual sensor reading or computed output
    """
    return {
        "ts": now_ms(),
        "device_id": device_id,
        "source": source,
        "data": data,
    }
