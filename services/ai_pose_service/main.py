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

    # ── Sticky risky state machine ────────────────────────────────────────────
    # The model resets is_risky on a single high-confidence frame.
    # We require SAFE_FRAMES_TO_CLEAR consecutive safe frames before clearing.
    _risky_state      = False
    _safe_frame_count = 0
    SAFE_FRAMES_TO_CLEAR = 10   # ~10s at 1 FPS to clear risky state
    NOSE_RISKY_THRESHOLD = 0.50

    # ── on_result callback — publishes every frame result to MQTT ──────────────
    def on_result(result: Dict, _frame: np.ndarray) -> None:
        nonlocal _risky_state, _safe_frame_count

        sleep_info   = result.get("sleep", {})
        blanket_info = result.get("blanket", {})
        nose_conf    = result.get("nose_confidence") or 0.0

        # Sticky state: go risky immediately on low confidence,
        # but only clear after SAFE_FRAMES_TO_CLEAR consecutive safe frames
        if nose_conf < NOSE_RISKY_THRESHOLD or result.get("is_risky", False):
            _risky_state      = True
            _safe_frame_count = 0
        else:
            _safe_frame_count += 1
            if _safe_frame_count >= SAFE_FRAMES_TO_CLEAR:
                _risky_state      = False
                _safe_frame_count = 0

        payload = make_message(
            source="ai_pose_service",
            data={
                "device_id":        device_id,
                "nose_confidence":  result.get("nose_confidence"),
                "is_risky":         _risky_state,
                "baby_state":       sleep_info.get("baby_state"),
                "ear":              sleep_info.get("ear"),
                "blanket_flag":     blanket_info.get("blanket_flag", False),
                "burst_activated":  result.get("burst_activated", False),
                "burst_false_alarm":result.get("burst_false_alarm", False),
            },
        )

        client.publish_json(pose_topic, payload, qos=1, retain=False)

        # Console summary
        nose   = result.get("nose_confidence")
        status = "RISKY" if _risky_state else "SAFE"
        state  = sleep_info.get("baby_state", "unknown")
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

    import io, contextlib
    try:
        print(f"[ai_pose] pipeline starting — camera: {ai_cfg['camera_url']}")
        # Suppress internal pipeline prints ([WARNING], [INFO], [BLANKET] etc.)
        with contextlib.redirect_stdout(io.StringIO()):
            run_pipeline(frame_source, config, on_result=on_result)
    except KeyboardInterrupt:
        print("[ai_pose] stopped.")
    finally:
        cleanup()
        client.close()


if __name__ == "__main__":
    main()
