"""
sleep_detection_service
───────────────────────
Polls the camera snapshot, uses MediaPipe FaceMesh to compute
Eye Aspect Ratio (EAR), and maintains a sleep/awake state machine.

State changes printed to terminal:
  [sleep] Baby fell asleep  (EAR=0.142)
  [sleep] Baby woke up      (EAR=0.312)

Auto-installs mediapipe on first run if missing.
No MQTT yet — print only.
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

POLL_INTERVAL_S          = 0.5
EAR_CLOSED_THRESHOLD     = 0.21
CLOSED_SECONDS_THRESHOLD = 10.0
OPEN_CONFIRM_SECONDS     = 0.5

LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ─────────────────────────────────────────────
#  Auto-install mediapipe
# ─────────────────────────────────────────────

def ensure_mediapipe():
    try:
        import mediapipe  # noqa
        return
    except ImportError:
        pass
    print("[sleep] Installing mediapipe (first run only)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--break-system-packages", "mediapipe"],
        check=True,
    )
    print("[sleep] mediapipe installed")


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_snapshot(url: str):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read()
        arr   = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        return frame
    except Exception:
        return None


def eye_aspect_ratio(pts: np.ndarray) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h  = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    ensure_mediapipe()
    import mediapipe as mp

    ai_cfg     = load_yaml("config/ai.yaml")["ai"]
    camera_url = ai_cfg.get("camera_url", "http://localhost:8001/snapshot")

    print(f"[sleep] starting — camera: {camera_url}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=True,
    )

    print("[sleep] FaceMesh ready — monitoring eye state")

    baby_state         = "awake"
    closed_start_time  = None
    open_confirm_start = None

    while True:
        try:
            frame = fetch_snapshot(camera_url)
            if frame is None:
                time.sleep(POLL_INTERVAL_S)
                continue

            h, w    = frame.shape[:2]
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            ear         = None
            eyes_closed = False

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                left_eye = np.array(
                    [[lm[i].x * w, lm[i].y * h] for i in LEFT_EYE_IDX],
                    dtype=np.float32,
                )
                right_eye = np.array(
                    [[lm[i].x * w, lm[i].y * h] for i in RIGHT_EYE_IDX],
                    dtype=np.float32,
                )

                ear         = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                eyes_closed = ear < EAR_CLOSED_THRESHOLD

            # ── State machine ──────────────────────────────────────────────
            t = time.time()

            if eyes_closed:
                open_confirm_start = None
                if closed_start_time is None:
                    closed_start_time = t
                if baby_state == "awake" and (t - closed_start_time) >= CLOSED_SECONDS_THRESHOLD:
                    baby_state = "asleep"
                    print(f"[sleep] Baby fell asleep  (EAR={ear:.3f})")
            else:
                closed_start_time = None
                if baby_state == "asleep":
                    if open_confirm_start is None:
                        open_confirm_start = t
                    if (t - open_confirm_start) >= OPEN_CONFIRM_SECONDS:
                        baby_state = "awake"
                        open_confirm_start = None
                        print(f"[sleep] Baby woke up  (EAR={ear:.3f if ear else 'N/A'})")

        except Exception as e:
            print(f"[sleep] ERROR: {e}")

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
