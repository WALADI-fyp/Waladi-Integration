"""
ai_pose_service — runs the infant pose/sleep detection pipeline and
publishes every result to MQTT on the waladi/ai/pose topic.

Reads config from:
  config/mqtt.yaml   — broker connection
  config/topics.yaml — topic names
  config/ai.yaml     — model path, camera URL, fps, etc.
"""

import json
import sys
import time
import yaml
from pathlib import Path
from typing import Dict

import numpy as np
import cv2

from config.device import get_device_id
from shared.mqtt_client import MqttClient
from shared.message import make_message

# Add project root to path so the standalone script can import cleanly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.ai_pose_service.standalone_pi_pose_ncnn import (
    PipelineConfig,
    FrameProcessingConfig,
    RiskDetectionConfig,
    BlanketDetectionConfig,
    SleepDetectionConfig,
    EndpointSourceConfig,
    build_frame_source,
    run_pipeline,
)

device_id = get_device_id()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    mqtt_cfg  = load_yaml("config/mqtt.yaml")
    topics    = load_yaml("config/topics.yaml")["topics"]
    ai_cfg    = load_yaml("config/ai.yaml")["ai"]

    pose_topic = topics["ai_pose"]

    # ── MQTT client ────────────────────────────────────────────────────────────
    client = MqttClient(
        client_id=f"ai_pose_{device_id}",
        host=mqtt_cfg["broker"]["host"],
        port=mqtt_cfg["broker"]["port"],
        keepalive=mqtt_cfg["client"]["keepalive"],
        username=mqtt_cfg["broker"]["username"],
        password=mqtt_cfg["broker"]["password"],
        tls=mqtt_cfg["broker"].get("tls", False),
    )
    client.connect()
    print(f"[ai_pose] MQTT connected, publishing to '{pose_topic}'")

    # ── Pipeline config ────────────────────────────────────────────────────────
    target_fps = ai_cfg.get("target_fps", 1)

    config = PipelineConfig(
        ncnn_model_dir=ai_cfg["model_dir"],
        output_dir=ai_cfg.get("output_dir", "./ai_output"),
        source_type="endpoint",
        frame_processing=FrameProcessingConfig(
            target_fps=target_fps,
            color_order="BGR",
        ),
        risk_detection=RiskDetectionConfig(
            normal_fps=target_fps,
        ),
        blanket_detection=BlanketDetectionConfig(),
        sleep_detection=SleepDetectionConfig(
            enabled=ai_cfg.get("sleep_detection", True),
            models_dir=ai_cfg.get("sleep_models_dir", "./models/pfld"),
        ),
        endpoint_source=EndpointSourceConfig(
            url=ai_cfg["camera_url"],
            mode=ai_cfg.get("source_mode", "snapshot"),
        ),
    )

    # ── Load YuNet face detector for multi-stage risk assessment ─────────────
    _yunet_path = str(
        Path(__file__).resolve().parent.parent /
        "sleep_detection_service" / "models" / "face_detection_yunet_2023mar.onnx"
    )
    _yunet = None
    if Path(_yunet_path).exists():
        _yunet = cv2.FaceDetectorYN.create(
            model=_yunet_path, config="",
            input_size=(640, 360),
            score_threshold=0.5, nms_threshold=0.3, top_k=1,
        )
        print("[ai_pose] YuNet face detector loaded for multi-stage risk")
    else:
        print(f"[ai_pose] WARNING: YuNet not found at {_yunet_path} — falling back to nose-conf only")

    # ── Multi-stage state machine ──────────────────────────────────────────────
    # Stage 1 (primary):   YuNet face detection  — face visible = safe
    # Stage 2 (secondary): YOLO nose confidence  — diagnostic only
    # Confirmation: 2 consecutive frames needed to flip state either way
    _risky_state        = False
    _unsafe_frame_count = 0
    _safe_frame_count   = 0
    CONFIRM_FRAMES      = 2     # frames needed to confirm a state change

    def _analyse_face(frame: np.ndarray) -> dict:
        """
        Run YuNet on the frame and return a face analysis dict:
          face_found    : bool — any face detected
          right_eye_vis : bool — right eye landmark is valid
          left_eye_vis  : bool — left eye landmark is valid
          eyes_visible  : int  — 0, 1, or 2 eyes visible

        YuNet landmark format per detection row (15 values):
          [0:4]  bounding box x,y,w,h
          [4:6]  right eye x,y
          [6:8]  left eye x,y
          [8:10] nose tip x,y
          [14]   confidence score
        Eye landmark is (0,0) or negative when not detected.
        """
        result = {"face_found": False, "right_eye_vis": False,
                  "left_eye_vis": False, "eyes_visible": 0}
        if _yunet is None or frame is None:
            return result

        h, w = frame.shape[:2]
        scale = min(640 / w, 360 / h, 1.0)
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        sh, sw = small.shape[:2]
        _yunet.setInputSize((sw, sh))
        _, faces = _yunet.detect(small)

        if faces is None or len(faces) == 0:
            return result

        result["face_found"] = True
        f = faces[0]
        # Eye landmarks — valid when both x and y are positive
        re_x, re_y = float(f[4]), float(f[5])
        le_x, le_y = float(f[6]), float(f[7])
        result["right_eye_vis"] = re_x > 1.0 and re_y > 1.0
        result["left_eye_vis"]  = le_x > 1.0 and le_y > 1.0
        result["eyes_visible"]  = int(result["right_eye_vis"]) + int(result["left_eye_vis"])
        return result

    # ── on_result callback — publishes every frame result to MQTT ──────────────
    def on_result(result: Dict, _frame: np.ndarray) -> None:
        nonlocal _risky_state, _unsafe_frame_count, _safe_frame_count

        sleep_info   = result.get("sleep", {})
        blanket_info = result.get("blanket", {})
        nose_conf    = result.get("nose_confidence") or 0.0

        # ── Multi-stage face + eye analysis ───────────────────────────────────
        fa = _analyse_face(_frame)
        face_found   = fa["face_found"]
        eyes_visible = fa["eyes_visible"]   # 0, 1, or 2

        # Combined safety logic — SAFE if EITHER signal confirms face is up:
        #   nose_safe: YOLO nose confidence is high → nose clearly visible
        #   face_safe: YuNet sees a face AND at least 1 eye is visible
        # RISKY only when BOTH signals fail simultaneously.
        NOSE_CONF_SAFE = 0.50
        nose_safe = nose_conf >= NOSE_CONF_SAFE
        face_safe = face_found and eyes_visible >= 1
        frame_safe = nose_safe or face_safe

        if frame_safe:
            _unsafe_frame_count = 0
            _safe_frame_count  += 1
            if _safe_frame_count >= CONFIRM_FRAMES:
                if _risky_state:
                    reason = []
                    if nose_safe: reason.append(f"nose_conf={nose_conf:.3f}")
                    if face_safe: reason.append(f"{eyes_visible} eye(s) visible")
                    print(f"[ai_pose] SAFE — {' + '.join(reason)}")
                _risky_state = False
        else:
            _safe_frame_count   = 0
            _unsafe_frame_count += 1
            if _unsafe_frame_count >= CONFIRM_FRAMES:
                if not _risky_state:
                    print(
                        f"[ai_pose] RISKY — nose_conf={nose_conf:.3f} "
                        f"face={face_found} eyes={eyes_visible} "
                        f"(both signals failed)"
                    )
                _risky_state = True

        payload = make_message(
            source="ai_pose_service",
            data={
                "device_id":        device_id,
                "nose_confidence":  result.get("nose_confidence"),
                "face_found":       face_found,
                "eyes_visible":     eyes_visible,
                "is_risky":         _risky_state,
                "baby_state":       sleep_info.get("baby_state"),
                "ear":              sleep_info.get("ear"),
                "blanket_flag":     blanket_info.get("blanket_flag", False),
                "burst_activated":  result.get("burst_activated", False),
                "burst_false_alarm":result.get("burst_false_alarm", False),
            },
        )

        client.publish_json(pose_topic, payload, qos=1, retain=False)

        # Console — every frame
        status    = "RISKY" if _risky_state else "SAFE "
        nose_ok   = "✓" if nose_conf >= 0.50 else "✗"
        face_ok   = "✓" if face_safe else "✗"
        frame_info = f"{_frame.shape}" if _frame is not None else "None"
        print(
            f"[ai_pose] {status} | "
            f"nose={nose_conf:.3f}{nose_ok} "
            f"face={face_found}{face_ok} eyes={eyes_visible} | "
            f"frame={frame_info} | "
            f"unsafe={_unsafe_frame_count} safe={_safe_frame_count}"
        )
    # ── Build frame source and run ─────────────────────────────────────────────
    class _FakeArgs:
        """Minimal args object so build_frame_source works without argparse."""
        endpoint_url      = ai_cfg["camera_url"]
        endpoint_mode     = ai_cfg.get("source_mode", "snapshot")
        endpoint_timeout  = 5.0
        endpoint_poll_interval = None
        endpoint_retry_delay   = 0.5
        endpoint_max_errors    = 10
        insecure          = False
        endpoint_header   = []
        endpoint_auth_user     = None
        endpoint_auth_password = None
        source            = None
        color_order       = None
        convert_to_bgr    = None

    frame_source, cleanup, source_type = build_frame_source(_FakeArgs(), config)
    config.source_type = source_type

    try:
        print(f"[ai_pose] pipeline starting — camera: {ai_cfg['camera_url']}")
        run_pipeline(frame_source, config, on_result=on_result)
    except KeyboardInterrupt:
        print("[ai_pose] stopped.")
    finally:
        cleanup()
        client.close()


if __name__ == "__main__":
    main()
