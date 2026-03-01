"""
camera_service — MJPEG stream + single-frame snapshot on http://<pi-ip>:8001

Endpoints:
    GET /status    → {"camera_available": true/false}
    GET /snapshot  → single JPEG frame  (use this in React Native)
    GET /video     → multipart MJPEG    (browser / VLC / curl)

Uses picamera2 with Pi Camera 3 NoIR config:
  • 2304×1296, continuous autofocus, full sensor crop
"""

import threading
import time
import logging

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, Response

from picamera2 import Picamera2
from libcamera import controls as libcontrols

log = logging.getLogger("camera_service")
logging.basicConfig(level=logging.INFO, format="[camera] %(message)s")

app = FastAPI()

# ── shared state ──────────────────────────────────────────────────────────────

_frame_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_camera_available = False


# ── capture loop (background thread) ─────────────────────────────────────────

def _capture_loop():
    global _latest_jpeg, _camera_available

    try:
        picam2 = Picamera2()

        config = picam2.create_preview_configuration(
            main={"size": (2304, 1296), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.5)  # let sensor settle

        # Continuous autofocus (Pi Camera 3 / 3 NoIR)
        picam2.set_controls({"AfMode": libcontrols.AfModeEnum.Continuous})

        # Full sensor field of view
        props = picam2.camera_properties
        if "ScalerCropMaximum" in props:
            picam2.set_controls({"ScalerCrop": props["ScalerCropMaximum"]})

        _camera_available = True
        log.info("Pi Camera 3 NoIR started — 2304×1296, continuous AF")

        while True:
            frame = picam2.capture_array("main")           # RGB888 numpy array
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            with _frame_lock:
                _latest_jpeg = buf.tobytes()

    except Exception as exc:
        log.error(f"Camera error: {exc}")
        _camera_available = False


threading.Thread(target=_capture_loop, daemon=True, name="capture").start()


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    return {"camera_available": _camera_available}


@app.get("/snapshot")
def snapshot():
    """Single JPEG — poll this from React Native at your desired fps."""
    with _frame_lock:
        jpeg = _latest_jpeg

    if jpeg is None:
        return JSONResponse({"error": "no frame yet"}, status_code=503)

    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/video")
def video():
    """Multipart MJPEG — works in browsers, VLC, curl, ffmpeg."""
    if not _camera_available and _latest_jpeg is None:
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
            "X-Accel-Buffering": "no",
        },
    )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    main()
