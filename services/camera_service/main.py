"""
camera_service — MJPEG video stream on http://<pi-ip>:8001/video

Uses picamera2 (Pi Camera 3 NoIR) with continuous autofocus at full sensor width.
Falls back gracefully to a 503 if the camera cannot be initialised so the
rest of the Waladi backend keeps running.

Usage:
    curl http://<pi-ip>:8001/video          # raw MJPEG
    <img src="http://<pi-ip>:8001/video">  # browser / HTML
    AVPlayer / VLC                          # iOS / desktop
"""

import threading
import time
import logging

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

import base64
import yaml

from shared.mqtt_client import MqttClient
from shared.message import make_message
from config.device import get_device_id

DEVICE_ID: str = get_device_id()

log = logging.getLogger("camera_service")
logging.basicConfig(level=logging.INFO, format="[camera] %(message)s")

app = FastAPI()

def _load_yaml(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

# Configs (safe defaults if missing)
_mqtt_cfg = _load_yaml('config/mqtt.yaml')
_topics_cfg = _load_yaml('config/topics.yaml').get('topics', {})
_camera_cfg = _load_yaml('config/camera.yaml').get('mqtt_snapshots', {})

_MQTT_SNAPSHOTS_ENABLED = bool(_camera_cfg.get('enabled', False))
_MQTT_SNAPSHOT_INTERVAL = float(_camera_cfg.get('interval_sec', 2.0))
_MQTT_SNAPSHOT_MAX_WIDTH = int(_camera_cfg.get('max_width', 640))
_MQTT_SNAPSHOT_JPEG_QUALITY = int(_camera_cfg.get('jpeg_quality', 60))

# ── camera state ──────────────────────────────────────────────────────────────

_frame_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_latest_jpeg_mqtt: bytes | None = None
_latest_mqtt_meta: dict | None = None
_camera_available = False


def _capture_loop():
    """Background thread: open Pi Camera 3 NoIR via picamera2 and encode frames."""
    global _latest_jpeg, _camera_available

    try:
        from picamera2 import Picamera2
        from libcamera import controls as libcontrols
    except ImportError:
        log.error("picamera2 / libcamera not found — install them on the Pi")
        return

    try:
        picam2 = Picamera2()

        config = picam2.create_preview_configuration(
            main={"size": (2304, 1296), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.5)  # allow sensor to settle

        # Continuous autofocus (Pi Camera 3 / 3 NoIR)
        picam2.set_controls({"AfMode": libcontrols.AfModeEnum.Continuous})

        # Full sensor field of view
        props = picam2.camera_properties
        if "ScalerCropMaximum" in props:
            picam2.set_controls({"ScalerCrop": props["ScalerCropMaximum"]})

        _camera_available = True
        log.info("Pi Camera 3 NoIR started — streaming at 2304×1296")

        while True:
            frame = picam2.capture_array("main")          # RGB888 numpy array
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # Full-res JPEG for HTTP endpoints
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )

            # Smaller JPEG for MQTT snapshots (optional)
            mqtt_jpeg = None
            mqtt_meta = None
            if _MQTT_SNAPSHOTS_ENABLED:
                h, w = frame.shape[:2]
                if w > _MQTT_SNAPSHOT_MAX_WIDTH and _MQTT_SNAPSHOT_MAX_WIDTH > 0:
                    scale = _MQTT_SNAPSHOT_MAX_WIDTH / float(w)
                    new_w = _MQTT_SNAPSHOT_MAX_WIDTH
                    new_h = max(1, int(h * scale))
                    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    small = frame
                    new_h, new_w = h, w

                ok2, buf2 = cv2.imencode(
                    ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, _MQTT_SNAPSHOT_JPEG_QUALITY]
                )
                if ok2:
                    mqtt_jpeg = buf2.tobytes()
                    mqtt_meta = {"w": int(new_w), "h": int(new_h), "bytes": int(len(mqtt_jpeg))}

            with _frame_lock:
                _latest_jpeg = buf.tobytes()
                global _latest_jpeg_mqtt, _latest_mqtt_meta
                if mqtt_jpeg is not None:
                    _latest_jpeg_mqtt = mqtt_jpeg
                    _latest_mqtt_meta = mqtt_meta


    except Exception as exc:
        log.error(f"Camera init/capture failed: {exc}")
        _camera_available = False


# Start capture immediately so the first client gets a frame fast
threading.Thread(target=_capture_loop, daemon=True, name="capture").start()


# ── MQTT snapshot publisher ───────────────────────────────────────────────────

def _mqtt_snapshot_loop():
    if not _MQTT_SNAPSHOTS_ENABLED:
        return

    broker = _mqtt_cfg.get('broker', {})
    client_cfg = _mqtt_cfg.get('client', {})
    topic = _topics_cfg.get('camera_snapshot', 'camera/snapshot')

    mqtt = MqttClient(
        client_id=f"camera_{DEVICE_ID}",
        host=broker.get('host', 'localhost'),
        port=int(broker.get('port', 1883)),
        keepalive=int(client_cfg.get('keepalive', 30)),
        username=broker.get('username'),
        password=broker.get('password'),
        tls=bool(broker.get('tls', False)),
    )
    mqtt.connect()

    seq = 0
    try:
        while True:
            with _frame_lock:
                jpeg = _latest_jpeg_mqtt
                meta = _latest_mqtt_meta

            if jpeg is None:
                time.sleep(0.2)
                continue

            payload = make_message(
                source='camera_service',
                data={
                    'device_id': DEVICE_ID,
                    'seq': seq,
                    'content_type': 'image/jpeg',
                    'jpeg_b64': base64.b64encode(jpeg).decode('ascii'),
                    'meta': meta or {},
                },
            )
            mqtt.publish_json(topic, payload, qos=1, retain=False)
            seq += 1
            time.sleep(_MQTT_SNAPSHOT_INTERVAL)
    finally:
        mqtt.close()


threading.Thread(target=_mqtt_snapshot_loop, daemon=True, name='mqtt_snapshots').start()


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    return {"camera_available": _camera_available}


@app.get("/snapshot")
def snapshot():
    """
    Returns a single JPEG frame — ideal for React Native which does not
    support multipart MJPEG natively. Poll this endpoint at your desired fps.
    """
    with _frame_lock:
        jpeg = _latest_jpeg

    if jpeg is None:
        return JSONResponse({"error": "no frame yet"}, status_code=503)

    from fastapi.responses import Response
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/video")
def video():
    """
    Multipart MJPEG stream — ~15 fps, 2304×1296 native.

    Compatible with:
      • <img src="..."> in any browser
      • curl / ffmpeg / VLC
      • Most IP-camera viewer apps on iOS/Android
    """
    if not _camera_available:
        # Return 503 until the capture thread has a frame ready
        if _latest_jpeg is None:
            return JSONResponse({"error": "camera not available"}, status_code=503)

    def generate():
        while True:
            with _frame_lock:
                jpeg = _latest_jpeg

            if jpeg is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg +
                b"\r\n"
            )
            time.sleep(1 / 15)  # ~15 fps to clients

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # prevent nginx from buffering the stream
        },
    )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    main()
