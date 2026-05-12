"""
Waladi Debug Viewer
───────────────────
Usage:
    python3 debug_viewer.py              # no rotation
    python3 debug_viewer.py --rotate 90  # rotate 90° CW
    python3 debug_viewer.py --rotate 180
    python3 debug_viewer.py --rotate 270
    python3 debug_viewer.py --rotate 90 --save   # save to config/audio.yaml and exit

Try 90, 180, 270 until the baby face looks upright in the window.
Then run with --save to persist it.
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml

CAMERA_URL    = "http://localhost:8001/snapshot"
YUNET_PATH    = "services/sleep_detection_service/models/face_detection_yunet_2023mar.onnx"
LANDMARK_PATH = "services/sleep_detection_service/models/face_landmark.tflite"
AUDIO_CFG     = "config/audio.yaml"

EAR_CLOSED_THRESHOLD  = 0.21
LEFT_EYE_IDX          = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX         = [362, 385, 387, 263, 373, 380]
LANDMARK_INPUT_SIZE   = 192
LANDMARK_FACE_PADDING = 0.30

ROTATIONS = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


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


def load_yunet():
    det = cv2.FaceDetectorYN.create(
        model=YUNET_PATH, config="",
        input_size=(640, 360),
        score_threshold=0.5, nms_threshold=0.3, top_k=5,
    )
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
    except Exception:
        return None


def rotate_frame(frame, angle):
    if angle in ROTATIONS:
        return cv2.rotate(frame, ROTATIONS[angle])
    return frame


def put_text(img, text, pos, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def save_rotation(angle):
    with open(AUDIO_CFG) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("audio", {})["frame_rotation"] = angle
    with open(AUDIO_CFG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[viewer] Saved frame_rotation={angle} to {AUDIO_CFG}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotate", type=int, default=0,
                        choices=[0, 90, 180, 270],
                        help="Rotate frame: 0, 90, 180, or 270 degrees CW")
    parser.add_argument("--save", action="store_true",
                        help="Save rotation to config/audio.yaml and exit")
    args = parser.parse_args()

    if args.save:
        save_rotation(args.rotate)
        print("[viewer] Rotation saved. Restart driver to apply.")
        return

    print(f"[viewer] Starting with rotation={args.rotate}°")
    print(f"[viewer] Try: python3 debug_viewer.py --rotate 90")
    print(f"[viewer] Try: python3 debug_viewer.py --rotate 180")
    print(f"[viewer] Try: python3 debug_viewer.py --rotate 270")
    print(f"[viewer] Once correct: python3 debug_viewer.py --rotate {args.rotate} --save")
    print(f"[viewer] Press Ctrl+C to stop")

    yunet = load_yunet()
    interp, in_d, out_d, lm_idx = load_landmarker()

    cv2.namedWindow("Waladi Debug Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Waladi Debug Viewer", 640, 500)

    while True:
        raw = fetch_frame()
        if raw is None:
            time.sleep(0.5)
            continue

        frame = rotate_frame(raw, args.rotate)
        vis   = frame.copy()
        h, w  = vis.shape[:2]

        yunet.setInputSize((w, h))
        _, faces = yunet.detect(frame)

        face_found   = False
        eyes_visible = 0
        ear          = None

        if faces is not None and len(faces) > 0:
            face_found = True
            f = faces[0]
            fx, fy, fw_, fh_ = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            cv2.rectangle(vis, (fx, fy), (fx+fw_, fy+fh_), (0, 255, 0), 2)

            re_x, re_y = int(f[4]), int(f[5])
            le_x, le_y = int(f[6]), int(f[7])
            re_vis = re_x > 1 and re_y > 1
            le_vis = le_x > 1 and le_y > 1
            eyes_visible = int(re_vis) + int(le_vis)

            if re_vis:
                cv2.circle(vis, (re_x, re_y), 6, (0, 255, 255), -1)
                put_text(vis, "RE", (re_x+7, re_y-5), (0,255,255), 0.4, 1)
            if le_vis:
                cv2.circle(vis, (le_x, le_y), 6, (255, 255, 0), -1)
                put_text(vis, "LE", (le_x+7, le_y-5), (255,255,0), 0.4, 1)

            pad  = int(LANDMARK_FACE_PADDING * min(fw_, fh_))
            x1   = max(0, fx - pad)
            y1   = max(0, fy - pad)
            x2   = min(w, fx + fw_ + pad)
            y2   = min(h, fy + fh_ + pad)
            crop = frame[y1:y2, x1:x2]
            lm   = run_landmarker(interp, in_d, out_d, lm_idx, crop)

            if lm is not None:
                lm_full = lm.copy()
                lm_full[:, 0] += x1
                lm_full[:, 1] += y1
                for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    px, py = int(lm_full[idx, 0]), int(lm_full[idx, 1])
                    cv2.circle(vis, (px, py), 2, (0, 0, 255), -1)
                left_ear  = eye_aspect_ratio(lm_full[LEFT_EYE_IDX])
                right_ear = eye_aspect_ratio(lm_full[RIGHT_EYE_IDX])
                ear       = (left_ear + right_ear) / 2.0

        # Status bar
        bar_h = 105
        cv2.rectangle(vis, (0, h - bar_h), (w, h), (0, 0, 0), -1)

        is_risky   = eyes_visible == 0
        risk_color = (0, 0, 255) if is_risky else (0, 255, 0)
        ear_str    = f"{ear:.3f}" if ear is not None else "N/A"
        closed     = ear is not None and ear < EAR_CLOSED_THRESHOLD

        put_text(vis, f"Risk: {'RISKY' if is_risky else 'SAFE'}",
                 (10, h - bar_h + 28), risk_color, 0.8, 2)
        put_text(vis, f"EAR: {ear_str}  ({'CLOSED' if closed else 'open'})  threshold < {EAR_CLOSED_THRESHOLD}",
                 (10, h - bar_h + 56), (255, 255, 255), 0.55, 1)
        put_text(vis, f"Face: {'YES' if face_found else 'NO'}  {eyes_visible}/2 eyes visible",
                 (10, h - bar_h + 80), (200, 200, 200), 0.55, 1)
        put_text(vis, f"Rotation: {args.rotate}deg  |  run with --rotate 90/180/270 to change",
                 (10, h - bar_h + 100), (100, 200, 255), 0.42, 1)

        cv2.imshow("Waladi Debug Viewer", vis)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
