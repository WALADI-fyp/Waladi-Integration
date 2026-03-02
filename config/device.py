import uuid
import pathlib

_ID_FILE = pathlib.Path(__file__).parent / "device_id.txt"


def get_device_id() -> str:
    """
    Returns a stable unique ID for this Pi.
    Generated once and persisted to config/device_id.txt.
    """
    if _ID_FILE.exists():
        return _ID_FILE.read_text().strip()
    device_id = f"waladi-{uuid.uuid4().hex[:8]}"
    _ID_FILE.write_text(device_id)
    print(f"[device] generated new device_id: {device_id}")
    return device_id
