"""
sleep_detection_service
───────────────────────
Polls the camera snapshot endpoint, runs YuNet face detection +
MediaPipe face_landmark.tflite eye-aspect-ratio (EAR) analysis,
and maintains a sleep/awake state machine.

State changes are printed to the terminal:
  [sleep] Baby fell asleep
  [sleep] Baby woke up

Models (bundled alongside this file):
  models/face_detection_yunet_2023mar.onnx
  models/face_landmark.tflite

No MQTT yet — print only.

Run from project root:
  python3 -m services.sleep_detection_service.main
"""

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

MODELS_DIR = Path(__file__).resolve().parent / "models"
YUNET_PATH    = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
LANDMARK_PATH = str(MODELS_DIR / "face_landmark.tflite")

POLL_INTERVAL_S       = 0.5     # how often to grab a snapshot
LANDMARK_INPUT_SIZE   = 192
LANDMARK_FACE_PADDING = 0.25
LANDMARK_MIN_FLAG     = 0.5
EAR_CLOSED_THRESHOLD  = 0.21
CLOSED_SECONDS_THRESHOLD = 10.0  # eyes closed this long → asleep
OPEN_CONFIRM_SECONDS  = 0.5      # eyes open this long → awake

# MediaPipe face landmark eye indices
LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ─────────────────────────────────────────────
#  TFLite interpreter
# ─────────────────────────────────────────────

def _get_interpreter_class():
    for mod, cls in [
        ("tflite_runtime.interpreter", "Interpreter"),
        ("ai_edge_litert.interpreter",  "Interpreter"),
        ("tensorflow.lite",             "Interpreter"),
    ]:
        try:
            m = __import__(mod, fromlist=[cls])
            return getattr(m, cls)
        except ImportError:
            continue
    raise RuntimeError(
        "No TFLite interpreter found.\n"
        "  pip install tflite-runtime --break-system-packages"
    )


# ─────────────────────────────────────────────
#  Model loaders
# ─────────────────────────────────────────────

def load_yunet() -> cv2.FaceDetectorYN:
    p = Path(YUNET_PATH)
    if not p.exists():
        raise FileNotFoundError(f"[sleep] YuNet model not found: {p}")
    det = cv2.FaceDetectorYN.create(
        model=str(p), config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=1,
    )
    print(f"[sleep] YuNet loaded")
    return det


def load_landmarker():
    p = Path(LANDMARK_PATH)
    if not p.exists():
        raise FileNotFoundError(f"[sleep] Landmark model not found: {p}")
    Interpreter = _get_interpreter_class()
    interp = Interpreter(model_path=str(p))
    interp.allocate_tensors()
    in_d  = interp.get_input_details()
    out_d = interp.get_output_details()

    lm_idx, flag_idx = None, None
    for i, d in enumerate(out_d):
        n = int(np.prod(d["shape"]))
        if n == 468 * 3:
            lm_idx = i
        elif n == 1:
            flag_idx = i

    if lm_idx is None:
        raise RuntimeError(f"[sleep] Could not find 1404-element landmark output in {p}")

    print(f"[sleep] Landmark model loaded")
    return interp, in_d, out_d, lm_idx, flag_idx


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_snapshot(url: str) -> "np.ndarray | None":
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read()
        arr   = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        # Resize to max 640px wide — YuNet works on 320x320 patches,
        # the full 2304×1296 makes faces appear tiny and undetectable
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        return frame
    except Exception:
        return None


def detect_face(yunet, frame: np.ndarray):
    fh, fw = frame.shape[:2]
    yunet.setInputSize((fw, fh))
    _, faces = yunet.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    f = faces[0]
    return int(f[0]), int(f[1]), int(f[2]), int(f[3])


def run_landmarker(interp, in_d, out_d, lm_idx, flag_idx, face_crop):
    ch, cw = face_crop.shape[:2]
    if ch == 0 or cw == 0:
        return None, 0.0

    rgb     = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE))
    inp     = (resized.astype(np.float32) / 255.0)[np.newaxis, ...]

    interp.set_tensor(in_d[0]["index"], inp)
    interp.invoke()

    raw  = np.array(interp.get_tensor(out_d[lm_idx]["index"])).reshape(-1, 3)
    flag = 1.0
    if flag_idx is not None:
        fv = float(np.array(interp.get_tensor(out_d[flag_idx]["index"])).flatten()[0])
        flag = fv if 0.0 <= fv <= 1.0 else float(1.0 / (1.0 + np.exp(-fv)))

    xy = raw[:, :2].astype(np.float32)
    xy[:, 0] *= (cw / LANDMARK_INPUT_SIZE)
    xy[:, 1] *= (ch / LANDMARK_INPUT_SIZE)
    return xy, flag


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
    ai_cfg     = load_yaml("config/ai.yaml")["ai"]
    camera_url = ai_cfg.get("camera_url", "http://localhost:8001/snapshot")

    print(f"[sleep] starting — camera: {camera_url}")

    try:
        yunet = load_yunet()
        interp, in_d, out_d, lm_idx, flag_idx = load_landmarker()
    except Exception as e:
        print(f"[sleep] ERROR loading models: {e}")
        return

    print("[sleep] ready — monitoring eye state")

    baby_state         = "awake"
    closed_start_time  = None
    open_confirm_start = None

    while True:
        try:
            frame = fetch_snapshot(camera_url)
            if frame is None:
                time.sleep(POLL_INTERVAL_S)
                continue

            h, w = frame.shape[:2]
            eyes_closed = False
            ear         = None

            face = detect_face(yunet, frame)
            h2, w2 = frame.shape[:2]
            print(f"[sleep] frame={w2}x{h2} face={'detected' if face else 'not detected'} ear={round(ear,3) if ear else None}")
            if face is not None:
                x, y, fw_, fh_ = face
                pad = int(LANDMARK_FACE_PADDING * min(fw_, fh_))
                x1  = max(0, x - pad)
                y1  = max(0, y - pad)
                x2  = min(w, x + fw_ + pad)
                y2  = min(h, y + fh_ + pad)
                crop = frame[y1:y2, x1:x2]

                lm, flag = run_landmarker(interp, in_d, out_d, lm_idx, flag_idx, crop)

                if lm is not None and flag >= LANDMARK_MIN_FLAG:
                    lm[:, 0] += x1
                    lm[:, 1] += y1

                    left_ear  = eye_aspect_ratio(lm[LEFT_EYE_IDX])
                    right_ear = eye_aspect_ratio(lm[RIGHT_EYE_IDX])
                    ear       = (left_ear + right_ear) / 2.0
                    eyes_closed = ear < EAR_CLOSED_THRESHOLD

            # ── State machine ─────────────────────────────────────────────
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
