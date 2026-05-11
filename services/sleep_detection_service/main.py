"""
sleep_detection_service
───────────────────────
YuNet face detection + face_landmark.tflite → EAR → sleep state.
Uses only tflite-runtime and opencv — no mediapipe needed.

Models in models/ folder:
  face_detection_yunet_2023mar.onnx
  face_landmark.tflite
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

POLL_INTERVAL_S          = 0.5
EAR_CLOSED_THRESHOLD     = 0.21
CLOSED_SECONDS_THRESHOLD = 10.0
OPEN_CONFIRM_SECONDS     = 0.5
LANDMARK_INPUT_SIZE      = 192
LANDMARK_FACE_PADDING    = 0.30

LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ─────────────────────────────────────────────
#  TFLite
# ─────────────────────────────────────────────

def get_interpreter_class():
    for mod, cls in [
        ("tflite_runtime.interpreter", "Interpreter"),
        ("ai_edge_litert.interpreter",  "Interpreter"),
    ]:
        try:
            m = __import__(mod, fromlist=[cls])
            return getattr(m, cls)
        except ImportError:
            continue
    raise RuntimeError("No TFLite interpreter. pip install tflite-runtime --break-system-packages")


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_snapshot(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read()
        arr   = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        return frame
    except Exception as e:
        print(f"[sleep] snapshot fetch failed: {e}")
        return None


def eye_aspect_ratio(pts: np.ndarray) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h  = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0


def load_yunet():
    if not Path(YUNET_PATH).exists():
        raise FileNotFoundError(f"YuNet not found: {YUNET_PATH}")
    det = cv2.FaceDetectorYN.create(
        model=YUNET_PATH, config="",
        input_size=(320, 320),
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=1,
    )
    print("[sleep] YuNet loaded")
    return det


def load_landmarker():
    if not Path(LANDMARK_PATH).exists():
        raise FileNotFoundError(f"Landmark model not found: {LANDMARK_PATH}")
    Interpreter = get_interpreter_class()
    interp = Interpreter(model_path=LANDMARK_PATH)
    interp.allocate_tensors()
    in_d  = interp.get_input_details()
    out_d = interp.get_output_details()

    print(f"[sleep] Landmark outputs:")
    lm_idx = None
    for i, d in enumerate(out_d):
        n = int(np.prod(d["shape"]))
        print(f"  [{i}] shape={list(d['shape'])} n={n} name={d['name']}")
        if n == 468 * 3:
            lm_idx = i

    if lm_idx is None:
        # Fallback: pick the largest output
        lm_idx = max(range(len(out_d)), key=lambda i: int(np.prod(out_d[i]["shape"])))
        print(f"[sleep] 1404-elem output not found, using largest output idx={lm_idx}")
    else:
        print(f"[sleep] Landmark output idx={lm_idx}")

    return interp, in_d, out_d, lm_idx


def run_landmarker(interp, in_d, out_d, lm_idx, crop_bgr):
    ch, cw = crop_bgr.shape[:2]
    if ch < 10 or cw < 10:
        return None
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE))
    inp     = (resized.astype(np.float32) / 255.0)[np.newaxis, ...]
    interp.set_tensor(in_d[0]["index"], inp)
    interp.invoke()
    raw = np.array(interp.get_tensor(out_d[lm_idx]["index"])).reshape(-1, 3)
    xy  = raw[:, :2].astype(np.float32)
    # Scale from model coords to crop pixel coords
    xy[:, 0] *= (cw / LANDMARK_INPUT_SIZE)
    xy[:, 1] *= (ch / LANDMARK_INPUT_SIZE)
    return xy


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    ai_cfg     = load_yaml("config/ai.yaml")["ai"]
    camera_url = ai_cfg.get("camera_url", "http://localhost:8001/snapshot")

    print(f"[sleep] starting — camera: {camera_url}")

    yunet                          = load_yunet()
    interp, in_d, out_d, lm_idx   = load_landmarker()

    print("[sleep] ready — monitoring eye state")

    baby_state         = "awake"
    closed_start_time  = None
    open_confirm_start = None
    frame_count        = 0

    while True:
        try:
            frame = fetch_snapshot(camera_url)
            if frame is None:
                time.sleep(POLL_INTERVAL_S)
                continue

            frame_count += 1
            h, w = frame.shape[:2]

            # ── Face detection ────────────────────────────────────────────
            yunet.setInputSize((w, h))
            _, faces = yunet.detect(frame)

            ear         = None
            eyes_closed = False

            if faces is not None and len(faces) > 0:
                f = faces[0]
                fx, fy, fw_, fh_ = int(f[0]), int(f[1]), int(f[2]), int(f[3])

                # Padded crop
                pad = int(LANDMARK_FACE_PADDING * min(fw_, fh_))
                x1  = max(0, fx - pad)
                y1  = max(0, fy - pad)
                x2  = min(w, fx + fw_ + pad)
                y2  = min(h, fy + fh_ + pad)
                crop = frame[y1:y2, x1:x2]

                # ── Landmark inference ────────────────────────────────────
                lm = run_landmarker(interp, in_d, out_d, lm_idx, crop)

                if lm is not None:
                    # Shift landmark coords from crop space to full frame
                    lm_full = lm.copy()
                    lm_full[:, 0] += x1
                    lm_full[:, 1] += y1

                    left_ear  = eye_aspect_ratio(lm_full[LEFT_EYE_IDX])
                    right_ear = eye_aspect_ratio(lm_full[RIGHT_EYE_IDX])
                    ear       = (left_ear + right_ear) / 2.0
                    eyes_closed = ear < EAR_CLOSED_THRESHOLD

                    # Log every 10 frames so we can see EAR values
                    if frame_count % 10 == 0:
                        print(
                            f"[sleep] EAR={ear:.3f} "
                            f"(L={left_ear:.3f} R={right_ear:.3f}) "
                            f"closed={eyes_closed} "
                            f"state={baby_state}"
                        )
                else:
                    if frame_count % 20 == 0:
                        print(f"[sleep] face detected but landmarks returned None (crop={crop.shape})")
            else:
                if frame_count % 20 == 0:
                    print(f"[sleep] no face detected in frame")

            # ── State machine ─────────────────────────────────────────────
            t = time.time()

            if eyes_closed:
                open_confirm_start = None
                if closed_start_time is None:
                    closed_start_time = t
                closed_dur = t - closed_start_time
                if baby_state == "awake" and closed_dur >= CLOSED_SECONDS_THRESHOLD:
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
