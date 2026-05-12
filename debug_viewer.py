"""
Waladi Debug Viewer
───────────────────
Opens a live window on the Pi showing what the AI models see.
Draws: face box, eye landmarks, EAR, risk state, sleep state.

Run from project root (separately from driver.py):
    python3 debug_viewer.py

Press Q to quit.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_URL   = "http://localhost:8001/snapshot"
YUNET_PATH   = "services/sleep_detection_service/models/face_detection_yunet_2023mar.onnx"
LANDMARK_PATH = "services/sleep_detection_service/models/face_landmark.tflite"
POLL_FPS     = 2   # how many frames per second to fetch

EAR_CLOSED_THRESHOLD = 0.21
LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LANDMARK_INPUT_SIZE   = 192
LANDMARK_FACE_PADDING = 0.30


# ── TFLite ────────────────────────────────────────────────────────────────────
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
    raise RuntimeError("No TFLite interpreter found.")


# ── Models ────────────────────────────────────────────────────────────────────
def load_yunet():
    det = cv2.FaceDetectorYN.create(
        model=YUNET_PATH, config="",
        input_size=(640, 360),
        score_threshold=0.5, nms_threshold=0.3, top_k=5,
    )
    print("[viewer] YuNet loaded")
    return det


def load_landmarker():
    Interpreter = get_interpreter_class()
    interp = Interpreter(model_path=LANDMARK_PATH)
    interp.allocate_tensors()
    in_d  = interp.get_input_details()
    out_d = interp.get_output_details()
    lm_idx = next(
        i for i, d in enumerate(out_d)
        if int(np.prod(d["shape"])) == 468 * 3
    )
    return interp, in_d, out_d, lm_idx


def run_landmarker(interp, in_d, out_d, lm_idx, crop):
    ch, cw = crop.shape[:2]
    if ch < 10 or cw < 10:
        return None
    rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE))
    inp     = (resized.astype(np.float32) / 255.0)[np.newaxis, ...]
    interp.set_tensor(in_d[0]["index"], inp)
    interp.invoke()
    raw = np.array(interp.get_tensor(out_d[lm_idx]["index"])).reshape(-1, 3)
    xy  = raw[:, :2].copy()
    xy[:, 0] *= (cw / LANDMARK_INPUT_SIZE)
    xy[:, 1] *= (ch / LANDMARK_INPUT_SIZE)
    return xy


def eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h  = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0


# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch_frame():
    try:
        with urllib.request.urlopen(CAMERA_URL, timeout=3) as resp:
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
        print(f"[viewer] fetch failed: {e}")
        return None


# ── Draw helpers ──────────────────────────────────────────────────────────────
def put_text(img, text, pos, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    yunet = load_yunet()
    interp, in_d, out_d, lm_idx = load_landmarker()

    print("[viewer] starting — press Q in window to quit")
    cv2.namedWindow("Waladi Debug Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Waladi Debug Viewer", 640, 400)

    delay = int(1000 / POLL_FPS)

    while True:
        frame = fetch_frame()
        if frame is None:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            put_text(blank, "Waiting for camera...", (20, 180), (0, 200, 255))
            cv2.imshow("Waladi Debug Viewer", blank)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            continue

        vis = frame.copy()
        h, w = vis.shape[:2]

        # ── YuNet face detection ──────────────────────────────────────────────
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(frame)

        face_found   = False
        eyes_visible = 0
        ear          = None
        is_risky     = True

        if faces is not None and len(faces) > 0:
            face_found = True
            f = faces[0]
            fx, fy, fw_, fh_ = int(f[0]), int(f[1]), int(f[2]), int(f[3])

            # Draw face box
            cv2.rectangle(vis, (fx, fy), (fx+fw_, fy+fh_), (0, 255, 0), 2)

            # YuNet eye landmarks
            re_x, re_y = int(f[4]), int(f[5])
            le_x, le_y = int(f[6]), int(f[7])
            re_vis = re_x > 1 and re_y > 1
            le_vis = le_x > 1 and le_y > 1
            eyes_visible = int(re_vis) + int(le_vis)

            if re_vis:
                cv2.circle(vis, (re_x, re_y), 5, (0, 255, 255), -1)
                put_text(vis, "RE", (re_x+6, re_y-4), (0,255,255), 0.4, 1)
            if le_vis:
                cv2.circle(vis, (le_x, le_y), 5, (255, 255, 0), -1)
                put_text(vis, "LE", (le_x+6, le_y-4), (255,255,0), 0.4, 1)

            # TFLite landmarks for EAR
            pad = int(LANDMARK_FACE_PADDING * min(fw_, fh_))
            x1  = max(0, fx - pad)
            y1  = max(0, fy - pad)
            x2  = min(w, fx + fw_ + pad)
            y2  = min(h, fy + fh_ + pad)
            crop = frame[y1:y2, x1:x2]
            lm = run_landmarker(interp, in_d, out_d, lm_idx, crop)

            if lm is not None:
                lm_full = lm.copy()
                lm_full[:, 0] += x1
                lm_full[:, 1] += y1

                # Draw eye landmark points
                for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    px, py = int(lm_full[idx, 0]), int(lm_full[idx, 1])
                    cv2.circle(vis, (px, py), 2, (0, 0, 255), -1)

                left_ear  = eye_aspect_ratio(lm_full[LEFT_EYE_IDX])
                right_ear = eye_aspect_ratio(lm_full[RIGHT_EYE_IDX])
                ear       = (left_ear + right_ear) / 2.0

            # Risk: safe if at least 1 eye visible
            is_risky = eyes_visible == 0

        # ── Overlay status ────────────────────────────────────────────────────
        risk_color  = (0, 0, 255) if is_risky else (0, 255, 0)
        risk_label  = "RISKY" if is_risky else "SAFE"
        ear_str     = f"{ear:.3f}" if ear is not None else "N/A"
        eyes_str    = f"{eyes_visible}/2 eyes"

        # Background bar
        cv2.rectangle(vis, (0, h-90), (w, h), (0,0,0), -1)

        put_text(vis, f"Risk: {risk_label}", (10, h-65), risk_color, 0.7, 2)
        put_text(vis, f"EAR: {ear_str}  (closed if < {EAR_CLOSED_THRESHOLD})",
                 (10, h-40), (255,255,255), 0.55, 1)
        put_text(vis, f"Face: {'YES' if face_found else 'NO'}  {eyes_str}",
                 (10, h-15), (200,200,200), 0.55, 1)

        cv2.imshow("Waladi Debug Viewer", vis)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
