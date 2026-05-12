"""
Waladi Debug Viewer
───────────────────
Shows two panels side by side:
  LEFT  — camera snapshot with face/eye detection overlaid
  RIGHT — the exact frame ai_pose_service is analyzing

Usage:
    python3 debug_viewer.py              # no rotation
    python3 debug_viewer.py --rotate 90
    python3 debug_viewer.py --rotate 180
    python3 debug_viewer.py --rotate 270
    python3 debug_viewer.py --rotate 90 --save   # save rotation and exit
"""

import argparse
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
POSE_FRAME    = "/tmp/waladi_pose_frame.jpg"

EAR_CLOSED_THRESHOLD  = 0.32
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


def fetch_frame(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
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


def load_pose_frame():
    """Load the frame ai_pose_service is actually analyzing."""
    try:
        frame = cv2.imread(POSE_FRAME)
        return frame
    except Exception:
        return None


def rotate_frame(frame, angle):
    if angle in ROTATIONS:
        return cv2.rotate(frame, ROTATIONS[angle])
    return frame


def put_text(img, text, pos, color=(255, 255, 255), scale=0.55, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def analyse_frame(yunet, interp, in_d, out_d, lm_idx, frame):
    """Run face detection + EAR on a frame. Returns (vis, face_found, eyes_visible, ear)."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    face_found = False
    eyes_visible = 0
    ear = None

    yunet.setInputSize((w, h))
    _, faces = yunet.detect(frame)

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
            lm[:, 0] += x1
            lm[:, 1] += y1
            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                cv2.circle(vis, (int(lm[idx,0]), int(lm[idx,1])), 2, (0,0,255), -1)
            ear = (eye_aspect_ratio(lm[LEFT_EYE_IDX]) +
                   eye_aspect_ratio(lm[RIGHT_EYE_IDX])) / 2.0

    return vis, face_found, eyes_visible, ear


def add_status_bar(vis, label, face_found, eyes_visible, ear, extra=""):
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, h-85), (w, h), (0,0,0), -1)
    put_text(vis, label, (8, h-65), (100,200,255), 0.6, 2)
    ear_str = f"{ear:.3f}" if ear is not None else "N/A"
    closed  = ear is not None and ear > EAR_CLOSED_THRESHOLD  # inverted due to rotation
    put_text(vis, f"EAR: {ear_str} ({'CLOSED' if closed else 'open'})",
             (8, h-42), (255,255,255), 0.5, 1)
    put_text(vis, f"Face: {'YES' if face_found else 'NO'}  {eyes_visible}/2 eyes  {extra}",
             (8, h-18), (200,200,200), 0.5, 1)
    return vis


def save_rotation(angle):
    with open(AUDIO_CFG) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("audio", {})["frame_rotation"] = angle
    with open(AUDIO_CFG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[viewer] Saved frame_rotation={angle} to {AUDIO_CFG}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.save:
        save_rotation(args.rotate)
        print("[viewer] Saved. Restart driver to apply.")
        return

    print(f"[viewer] rotation={args.rotate}°  |  Ctrl+C to stop")
    print(f"[viewer] LEFT panel = camera snapshot with your rotation")
    print(f"[viewer] RIGHT panel = exact frame ai_pose is analyzing")

    yunet = load_yunet()
    interp, in_d, out_d, lm_idx = load_landmarker()

    PANEL_W = 500
    PANEL_H = 400
    GAP     = 10
    WIN_W   = PANEL_W * 2 + GAP * 3
    WIN_H   = PANEL_H + 10

    cv2.namedWindow("Waladi Debug Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Waladi Debug Viewer", WIN_W, WIN_H)

    blank = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    while True:
        canvas = np.full((WIN_H, WIN_W, 3), 30, dtype=np.uint8)

        # ── LEFT: camera snapshot with rotation ───────────────────────────────
        raw = fetch_frame(CAMERA_URL)
        if raw is not None:
            rotated = rotate_frame(raw, args.rotate)
            vis_l, ff_l, ev_l, ear_l = analyse_frame(
                yunet, interp, in_d, out_d, lm_idx, rotated)
            vis_l = add_status_bar(vis_l, f"SNAPSHOT rot={args.rotate}deg", ff_l, ev_l, ear_l)
            vis_l = cv2.resize(vis_l, (PANEL_W, PANEL_H))
        else:
            vis_l = blank.copy()
            put_text(vis_l, "Waiting for camera...", (20, PANEL_H//2), (0,200,255))

        canvas[5:5+PANEL_H, GAP:GAP+PANEL_W] = vis_l

        # ── RIGHT: frame ai_pose is actually analyzing ─────────────────────────
        pose_frame = load_pose_frame()
        if pose_frame is not None:
            vis_r, ff_r, ev_r, ear_r = analyse_frame(
                yunet, interp, in_d, out_d, lm_idx, pose_frame)
            age = time.time() - Path(POSE_FRAME).stat().st_mtime
            vis_r = add_status_bar(vis_r, "AI POSE FRAME", ff_r, ev_r, ear_r,
                                   extra=f"({age:.1f}s ago)")
            vis_r = cv2.resize(vis_r, (PANEL_W, PANEL_H))
        else:
            vis_r = blank.copy()
            put_text(vis_r, "No pose frame yet", (20, PANEL_H//2), (100,100,255))
            put_text(vis_r, "(driver.py must be running)", (20, PANEL_H//2+30),
                     (150,150,150), 0.45)

        canvas[5:5+PANEL_H, GAP*2+PANEL_W:GAP*2+PANEL_W*2] = vis_r

        # Labels
        put_text(canvas, "LEFT: Your view (rotated snapshot)",
                 (GAP, 18), (100,200,255), 0.5, 1)
        put_text(canvas, "RIGHT: What ai_pose model sees",
                 (GAP*2+PANEL_W, 18), (100,255,100), 0.5, 1)

        cv2.imshow("Waladi Debug Viewer", canvas)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
