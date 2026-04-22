#!/usr/bin/env python3
"""Standalone Raspberry Pi NCNN infant monitoring pipeline.

This file merges the logic previously split across:

- advanced_pose_infer/config.py
- advanced_pose_infer/frame_processing.py
- advanced_pose_infer/inference.py
- advanced_pose_infer/blanket_detection.py
- advanced_pose_infer/reporting.py
- advanced_pose_infer/pipeline.py
- ncnnBaby-sleep-detection/baby_monitor_ncnn.py  (sleep/eye detection)

It can be run directly on a Raspberry Pi without importing the local
package, and supports frames coming from an HTTP camera endpoint that
returns either:

- a single rendered image per request ("snapshot" mode), or
- an MJPEG multipart stream ("mjpeg" mode).

It processes frames at a configurable FPS (default 1 fps). When nose
confidence drops below the risk threshold, it temporarily bursts to
5× FPS for up to 10 seconds to quickly confirm or dismiss a risky pose.

Events are appended as JSON lines to a .log file in the output directory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUN GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL
-----
The infant-specific YOLOv26n-pose NCNN model is located at:

    ../infant-pose-model/pretrained_models/v26-model-set/
        v26n-daynight-150epoch_ncnn_model/

It was trained on day and night (IR) infant footage and outputs 17
COCO keypoints at 416×416. Pass the directory to --model; Ultralytics
requires the name to end in ``_ncnn_model``, which it already does.

DEPENDENCIES
------------
Install on the Pi (or dev machine) before running:

    pip install ncnn opencv-python-headless numpy requests ultralytics

The PFLD sleep-detection models (~1.6 MB total) are downloaded
automatically on first run into ./models/ unless --no-sleep-detection
is passed.

BASIC USAGE
-----------
Minimal run against an MJPEG stream (recommended — enables burst mode):

    python pi_simulation/standalone_pi_pose_ncnn.py \\
        --model ../infant-pose-model/pretrained_models/v26-model-set/v26n-daynight-150epoch_ncnn_model \\
        --url   http://<camera-ip>/video

Snapshot endpoint (one JPEG per HTTP GET, burst mode has no effect):

    python pi_simulation/standalone_pi_pose_ncnn.py \\
        --model ../infant-pose-model/pretrained_models/v26-model-set/v26n-daynight-150epoch_ncnn_model \\
        --url   http://<camera-ip>/snapshot \\
        --source-mode snapshot

Run without sleep detection (skips PFLD model download):

    python pi_simulation/standalone_pi_pose_ncnn.py \\
        --model ../infant-pose-model/pretrained_models/v26-model-set/v26n-daynight-150epoch_ncnn_model \\
        --url   http://<camera-ip>/video \\
        --no-sleep-detection

KEY OPTIONAL FLAGS
------------------
--target-fps INT          Processing rate in frames/sec (default: 1).
--burst-multiplier INT    FPS multiplier during burst mode (default: 5).
--burst-timeout FLOAT     Max seconds in burst before false-alarm (default: 10).
--output-dir PATH         Directory for saved frames and log (default: ./advanced_inference_output).
--log-file PATH           Explicit log file path (default: {output-dir}/monitor.log).
--status-log-interval F   Seconds between heartbeat status log entries (default: 30).
--max-seconds FLOAT       Stop pipeline after this many seconds.
--ear-threshold FLOAT     EAR below this = eyes closed (default: 0.21).
--sleep-seconds FLOAT     Seconds closed before baby is asleep (default: 10).
--sleep-models-dir PATH   Local dir for PFLD model files (default: ./models).
--benchmark               Collect timing stats and write a report on exit.
--insecure                Skip TLS certificate verification.
--header KEY=VALUE        Extra HTTP request header (repeatable).
--auth-user / --auth-password  Basic-auth credentials.

OUTPUT
------
Saved frames:   {output-dir}/risky_frame_<timestamp>.jpg   (on confirmed pose risk)
                {output-dir}/periodic_<timestamp>.jpg       (every --save-interval seconds)
Event log:      {output-dir}/monitor.log  (JSON lines, one event per line)

Example log entries:

    {"timestamp": "...", "type": "status",      "nose_confidence": 0.87, "baby_state": "asleep", "blanket_flag": false, "ear": 0.13}
    {"timestamp": "...", "type": "burst_start", "new_fps": 5}
    {"timestamp": "...", "type": "false_alarm", "reason": "nose_reappeared", "burst_duration_seconds": 2.1}
    {"timestamp": "...", "type": "pose_risk",   "nose_confidence": 0.21, "consecutive_low_conf": 5}
    {"timestamp": "...", "type": "blanket",     "invisible_count": 5, "invisible_names": ["left_hip", ...]}
    {"timestamp": "...", "type": "sleep_state", "new_state": "asleep", "ear": 0.14}
    {"timestamp": "...", "type": "kp_divergence", "keypoints": [{"name": "left_hip", "current": 0.12, "mean": 0.67, "delta": -0.55}]}

SIMULATION (no physical camera)
--------------------------------
A local video file or webcam can be used for testing via the Pi
simulation helper in pi_simulation/:

    python pi_simulation/standalone_pi_pose_ncnn.py \\
        --model ../infant-pose-model/pretrained_models/v26-model-set/v26n-daynight-150epoch_ncnn_model \\
        --url   http://localhost:5000/video
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Literal, Optional, Tuple

import cv2
import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover - handled at runtime on device
    requests = None

try:
    import ncnn
except ImportError:  # pragma: no cover - optional on dev machines
    ncnn = None


# ─────────────────────────────────────────────────────────────────────────────
# Small image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _frame_to_bgr(frame: np.ndarray, color_order: str) -> np.ndarray:
    """Convert frame to BGR if it arrives in RGB order; otherwise return unchanged."""
    if color_order.upper() == "RGB":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def _resize_frame(
    frame: np.ndarray,
    enabled: bool,
    dimensions: Tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    """Resize *frame* to *dimensions* (W, H) when *enabled*; otherwise pass through."""
    if not enabled:
        return frame
    width, height = dimensions
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def _prepare_overlay_frame(
    raw_frame: np.ndarray,
    frame_cfg: "FrameProcessingConfig",
) -> np.ndarray:
    """Apply color conversion and optional resize from a ``FrameProcessingConfig``."""
    frame = _frame_to_bgr(raw_frame, frame_cfg.color_order)
    interpolation = INTERPOLATION_MAP.get(
        frame_cfg.resize_interpolation.upper(),
        cv2.INTER_LINEAR,
    )
    return _resize_frame(
        frame,
        frame_cfg.resize_enabled,
        frame_cfg.resize_dimensions,
        interpolation,
    )


def _decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode a raw JPEG/PNG byte string into a BGR numpy array. Returns None on failure."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Sleep-detection utilities  (ported from baby_monitor_ncnn.py)
# ─────────────────────────────────────────────────────────────────────────────

def euclidean(a, b) -> float:
    """Return the Euclidean distance between two 2-D points *a* and *b*."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def eye_aspect_ratio(pts) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)"""
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4) + 1e-6)


def download_pfld_models(models_dir: str) -> None:
    """Download PFLD ncnn model files (pfld.param + pfld.bin) into *models_dir* if absent."""
    _PARAM_URL = "https://github.com/nihui/ncnn-assets/raw/master/models/pfld.param"
    _BIN_URL   = "https://github.com/nihui/ncnn-assets/raw/master/models/pfld.bin"
    os.makedirs(models_dir, exist_ok=True)
    for url, fname in [(_PARAM_URL, "pfld.param"), (_BIN_URL, "pfld.bin")]:
        path = os.path.join(models_dir, fname)
        if not os.path.exists(path):
            print(f"[SLEEP] Downloading {fname} ...")
            urllib.request.urlretrieve(url, path)
            print(f"[SLEEP]   → saved to {path}")
        else:
            print(f"[SLEEP] {fname} already present.")


def _parse_header_values(values: List[str]) -> Dict[str, str]:
    """Parse a list of ``KEY=VALUE`` or ``KEY:VALUE`` strings into a header dict.

    Raises ``ValueError`` if any entry is malformed or has an empty key.
    """
    headers: Dict[str, str] = {}
    for value in values:
        if "=" in value:
            key, header_value = value.split("=", 1)
        elif ":" in value:
            key, header_value = value.split(":", 1)
        else:
            raise ValueError(
                f"Invalid header '{value}'. Use KEY=VALUE or KEY:VALUE."
            )
        key = key.strip()
        header_value = header_value.strip()
        if not key:
            raise ValueError(f"Invalid empty header key in '{value}'.")
        headers[key] = header_value
    return headers


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameProcessingConfig:
    """Controls how raw frames are pre-processed before reaching the model.

    Attributes:
        max_frames: Stop after this many processed frames (None = unlimited).
        frame_skip: Process only every N-th frame seen by the source iterator.
        target_fps: Desired processing rate in frames per second.
        resize_enabled: Whether to resize frames to ``resize_dimensions``.
        resize_dimensions: Target (width, height) for model input.
        resize_interpolation: OpenCV interpolation name (LINEAR, NEAREST, …).
        color_order: Declares whether incoming frames are BGR or RGB.
        convert_to_bgr: Convert RGB frames to BGR before inference if True.
        normalize: Scale pixel values to [0, 1] (float32) if True.
    """

    max_frames: Optional[int] = None
    frame_skip: int = 1
    target_fps: int = 1
    resize_enabled: bool = True
    resize_dimensions: Tuple[int, int] = (416, 416)
    resize_interpolation: str = "LINEAR"
    color_order: Literal["RGB", "BGR"] = "BGR"
    convert_to_bgr: bool = False
    normalize: bool = False


@dataclass
class BlanketDetectionConfig:
    """Tuning knobs for lower-body occlusion (blanket) detection.

    Attributes:
        enabled: Master switch; set False to skip all blanket checks.
        lower_body_indices: COCO keypoint indices treated as lower-body
            (default: hips 11-12, knees 13-14, ankles 15-16).
        visibility_threshold: A keypoint is "invisible" when its confidence
            falls below this value.
        min_invisible_keypoints: Minimum number of invisible lower-body
            keypoints required to count a frame as occluded.
        consecutive_frames_to_flag: How many consecutive occluded frames
            must occur before raising a blanket alert.
        cooldown_seconds: Minimum time between successive blanket alerts.
    """

    enabled: bool = True
    lower_body_indices: Tuple[int, ...] = (11, 12, 13, 14, 15, 16)
    visibility_threshold: float = 0.30
    min_invisible_keypoints: int = 4
    consecutive_frames_to_flag: int = 5
    cooldown_seconds: float = 30.0


@dataclass
class SleepDetectionConfig:
    """Settings for the eye-state / sleep detection subsystem.

    Attributes:
        enabled: Master switch; set False (or pass ``--no-sleep-detection``)
            to skip PFLD model loading entirely.
        ear_closed_threshold: Eye Aspect Ratio below this value is treated
            as "eyes closed". Typical range 0.18 – 0.25.
        closed_seconds_threshold: Seconds eyes must be continuously closed
            before transitioning from "awake" → "asleep".
        open_confirm_seconds: Seconds eyes must be open after waking before
            the state transitions from "asleep" → "awake" (debounce).
        models_dir: Local directory where PFLD ncnn files are cached.
        ncnn_threads: Number of ncnn inference threads for the landmark model.
    """

    enabled: bool = True
    ear_closed_threshold: float = 0.21
    closed_seconds_threshold: float = 10.0
    open_confirm_seconds: float = 0.5
    models_dir: str = "./models"
    ncnn_threads: int = 2


@dataclass
class RiskDetectionConfig:
    """Pose risk and burst-mode tuning parameters.

    Attributes:
        risk_threshold: Nose keypoint confidence below this value is considered
            a low-confidence (potentially risky) frame.
        consecutive_risk_frames: How many consecutive low-confidence frames must
            occur before a confirmed risk alert is raised.
        confidence_history_size: Rolling-window size for per-keypoint confidence
            histories used by ``check_keypoint_divergence``.
        confidence_drop_margin: A keypoint is "diverging" when its current
            confidence is more than this amount below its rolling mean.
        monitoring_seconds: Seconds between monitoring checkpoints (priority 0).
        monitoring_frames: Frames between monitoring checkpoints (priority 1).
        priority_index: 0 = time-based monitoring; 1 = frame-count-based.
        save_interval_seconds: Minimum gap between periodic frame saves.
        burst_fps_multiplier: FPS is multiplied by this when a low-confidence
            frame is detected (burst mode). Effective on MJPEG sources.
        burst_timeout_seconds: Maximum time to stay in burst mode without
            confirming a risk; after this a false alarm is declared.
        normal_fps: Baseline FPS to return to after burst mode ends.
            Should match ``FrameProcessingConfig.target_fps``.
    """

    risk_threshold: float = 0.50
    consecutive_risk_frames: int = 5
    confidence_history_size: int = 10
    confidence_drop_margin: float = 0.15
    monitoring_seconds: float = 10.0
    monitoring_frames: int = 50
    priority_index: int = 0
    save_interval_seconds: float = 30.0
    # Burst-mode settings
    burst_fps_multiplier: int = 5
    burst_timeout_seconds: float = 10.0
    normal_fps: int = 1   # baseline FPS to return to after burst; should match target_fps


@dataclass
class EndpointSourceConfig:
    """Connection parameters for an HTTP camera endpoint.

    Attributes:
        url: Full URL of the endpoint (snapshot or MJPEG).
        mode: ``"auto"`` probes the content type; ``"snapshot"`` or
            ``"mjpeg"`` bypass auto-detection.
        timeout_seconds: Per-request / connection timeout.
        poll_interval_seconds: Minimum delay between snapshot requests
            (None → derived from ``target_fps``).
        retry_delay_seconds: Pause before retrying after a failed request.
        max_consecutive_errors: Stop iterating after this many consecutive
            failures. 0 means retry forever.
        verify_ssl: Set False (``--insecure``) to skip TLS verification.
        headers: Extra HTTP headers sent with every request.
        auth_user: Basic-auth username.
        auth_password: Basic-auth password.
    """

    url: str = ""
    mode: Literal["auto", "snapshot", "mjpeg"] = "auto"
    timeout_seconds: float = 5.0
    poll_interval_seconds: Optional[float] = None
    retry_delay_seconds: float = 0.5
    max_consecutive_errors: int = 10
    verify_ssl: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    auth_user: Optional[str] = None
    auth_password: Optional[str] = None


@dataclass
class PipelineConfig:
    """Top-level configuration for the full monitoring pipeline.

    Attributes:
        ncnn_model_dir: Path to the NCNN export directory containing
            ``model.ncnn.bin`` + ``model.ncnn.param``.
        imgsz: Model input image size passed to Ultralytics ``YOLO(...)``.
        output_dir: Directory for saved frames and benchmark reports.
        source_type: Filled by ``build_frame_source``; ``"endpoint"`` or
            ``"opencv"``. Used internally for burst-mode warnings.
        max_seconds: Stop the pipeline after this many seconds (None = run forever).
        benchmark: If True, collect timing stats and write a JSON report.
        report_json: Path for the JSON benchmark report (implies benchmark).
        report_md: Path for the Markdown benchmark report.
        status_log_interval_seconds: Minimum gap between ``status`` log entries.
        log_file: Explicit log file path. Defaults to
            ``{output_dir}/monitor.log``.
        frame_processing: Frame pre-processing settings.
        risk_detection: Pose risk and burst-mode settings.
        blanket_detection: Blanket occlusion detection settings.
        sleep_detection: Eye-state / sleep detection settings.
        endpoint_source: HTTP endpoint connection settings.
    """

    ncnn_model_dir: str = ""
    imgsz: int = 416
    output_dir: str = "./advanced_inference_output"
    source_type: str = "endpoint"
    max_seconds: Optional[float] = None
    benchmark: bool = False
    report_json: Optional[str] = None
    report_md: Optional[str] = None
    status_log_interval_seconds: float = 30.0
    log_file: Optional[str] = None

    frame_processing: FrameProcessingConfig = field(
        default_factory=FrameProcessingConfig,
    )
    risk_detection: RiskDetectionConfig = field(
        default_factory=RiskDetectionConfig,
    )
    blanket_detection: BlanketDetectionConfig = field(
        default_factory=BlanketDetectionConfig,
    )
    sleep_detection: SleepDetectionConfig = field(
        default_factory=SleepDetectionConfig,
    )
    endpoint_source: EndpointSourceConfig = field(
        default_factory=EndpointSourceConfig,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

INTERPOLATION_MAP = {
    "NEAREST": cv2.INTER_NEAREST,
    "LINEAR": cv2.INTER_LINEAR,
    "AREA": cv2.INTER_AREA,
    "CUBIC": cv2.INTER_CUBIC,
    "LANCZOS4": cv2.INTER_LANCZOS4,
}

COCO_KP_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


# ─────────────────────────────────────────────────────────────────────────────
# Frame processor
# ─────────────────────────────────────────────────────────────────────────────

class FrameProcessor:
    """Time-based frame gate that enforces the target FPS and applies pre-processing.

    The gate works by comparing the current wall-clock time to the last
    processed frame time. ``prepare()`` returns ``(False, None)`` for frames
    that arrive too early, effectively throttling the pipeline to
    ``target_fps``. ``set_target_fps()`` allows burst mode to update the rate
    at runtime without recreating the object.
    """

    def __init__(self, config: FrameProcessingConfig) -> None:
        """Initialise the processor and pre-compute the frame interval from ``target_fps``."""
        self.cfg = config
        self._frames_seen = 0
        self._frames_processed = 0
        self._last_process_time = 0.0
        self._frame_interval = 1.0 / max(config.target_fps, 1)
        self._interp = INTERPOLATION_MAP.get(
            config.resize_interpolation.upper(),
            cv2.INTER_LINEAR,
        )

    def set_target_fps(self, fps: int) -> None:
        """Dynamically change the processing rate (used by burst mode)."""
        self._frame_interval = 1.0 / max(fps, 1)

    def prepare(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Gate and pre-process one frame.

        Returns:
            ``(True, prepared_frame)`` when the frame should be processed.
            ``(False, None)`` when it should be skipped (rate-limit, frame-skip,
            or budget exhausted).
        """
        self._frames_seen += 1
        if self.cfg.max_frames is not None:
            if self._frames_processed >= self.cfg.max_frames:
                return False, None
        if self.cfg.frame_skip > 1:
            if (self._frames_seen - 1) % self.cfg.frame_skip != 0:
                return False, None
        now = time.time()
        if now - self._last_process_time < self._frame_interval:
            return False, None
        self._last_process_time = now

        prepared = frame
        if self.cfg.convert_to_bgr and self.cfg.color_order.upper() == "RGB":
            prepared = cv2.cvtColor(prepared, cv2.COLOR_RGB2BGR)
        prepared = _resize_frame(
            prepared,
            self.cfg.resize_enabled,
            self.cfg.resize_dimensions,
            self._interp,
        )
        if self.cfg.normalize:
            prepared = prepared.astype(np.float32) / 255.0
        self._frames_processed += 1
        return True, prepared

    def budget_exhausted(self) -> bool:
        """Return True when ``max_frames`` has been reached (always False if unlimited)."""
        if self.cfg.max_frames is None:
            return False
        return self._frames_processed >= self.cfg.max_frames

    @property
    def frames_seen(self) -> int:
        """Total frames received from the source (including skipped ones)."""
        return self._frames_seen

    @property
    def frames_processed(self) -> int:
        """Frames that passed the gate and were forwarded to the model."""
        return self._frames_processed


# ─────────────────────────────────────────────────────────────────────────────
# Blanket detector
# ─────────────────────────────────────────────────────────────────────────────

class BlanketDetector:
    """Detect sustained lower-body occlusion, typically caused by a blanket.

    The detector counts how many of the configured lower-body keypoints (hips,
    knees, ankles) fall below a visibility threshold in each frame. When that
    count exceeds ``min_invisible_keypoints`` for ``consecutive_frames_to_flag``
    consecutive frames, a blanket alert is raised and ``newly_raised`` is set to
    True in the returned result dict.  A cooldown timer prevents repeated alerts.
    """

    _KP_NAMES: Tuple[str, ...] = COCO_KP_NAMES

    def __init__(self, config: BlanketDetectionConfig) -> None:
        """Initialise state counters and an optional alert callback."""
        self.cfg = config
        self._consecutive_occluded = 0
        self._flag_active = False
        self._last_flag_time = 0.0
        self._total_flags_raised = 0
        self._flag_history: List[Dict] = []
        self._blanket_callback: Optional[Callable[[int, List[str]], None]] = None

    def update(self, keypoint_confidences: Optional[np.ndarray]) -> Dict:
        """Process one frame's keypoint confidences and return a status dict.

        Args:
            keypoint_confidences: Array of shape (17,) from YOLO, or None when
                no person was detected (treated as fully occluded).

        Returns:
            Dict with keys: ``blanket_flag``, ``invisible_count``,
            ``invisible_names``, ``consecutive``, ``newly_raised``.
        """
        if not self.cfg.enabled:
            return self._make_result(False, 0, [], 0, False)

        invisible_count, invisible_names = self._count_invisible(keypoint_confidences)
        is_occluded = invisible_count >= self.cfg.min_invisible_keypoints

        if is_occluded:
            self._consecutive_occluded += 1
        else:
            self._consecutive_occluded = 0
            self._flag_active = False

        newly_raised = False
        if (
            self._consecutive_occluded >= self.cfg.consecutive_frames_to_flag
            and not self._flag_active
        ):
            now = time.time()
            if now - self._last_flag_time >= self.cfg.cooldown_seconds:
                self._flag_active = True
                self._last_flag_time = now
                self._total_flags_raised += 1
                newly_raised = True
                self._record_flag(invisible_count, invisible_names)
                self._emit_blanket_alert(invisible_count, invisible_names)

        return self._make_result(
            self._flag_active,
            invisible_count,
            invisible_names,
            self._consecutive_occluded,
            newly_raised,
        )

    def set_blanket_callback(self, cb: Callable[[int, List[str]], None]) -> None:
        """Register a custom alert callback ``cb(invisible_count, invisible_names)``
        that fires instead of the default ``print`` when a blanket is detected."""
        self._blanket_callback = cb

    @property
    def total_flags_raised(self) -> int:
        """Cumulative number of blanket alerts raised since the detector was created."""
        return self._total_flags_raised

    @property
    def flag_history(self) -> List[Dict]:
        """Snapshot of all past blanket alert records (timestamp, invisible count, names)."""
        return list(self._flag_history)

    def _count_invisible(
        self,
        kp_confs: Optional[np.ndarray],
    ) -> Tuple[int, List[str]]:
        """Count lower-body keypoints below the visibility threshold.

        Returns:
            ``(count, names)`` — number of invisible keypoints and their names.
            When *kp_confs* is None (no detection), all lower-body keypoints
            are reported as invisible.
        """
        if kp_confs is None:
            names = [
                self._KP_NAMES[i]
                for i in self.cfg.lower_body_indices
                if i < len(self._KP_NAMES)
            ]
            return len(self.cfg.lower_body_indices), names
        invisible = 0
        names: List[str] = []
        for idx in self.cfg.lower_body_indices:
            if idx >= len(kp_confs):
                continue
            if kp_confs[idx] < self.cfg.visibility_threshold:
                invisible += 1
                if idx < len(self._KP_NAMES):
                    names.append(self._KP_NAMES[idx])
        return invisible, names

    def _record_flag(self, count: int, names: List[str]) -> None:
        """Append a blanket alert entry to the in-memory history."""
        self._flag_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "invisible_count": count,
                "invisible_keypoints": names,
                "consecutive_frames": self._consecutive_occluded,
            }
        )

    def _emit_blanket_alert(self, count: int, names: List[str]) -> None:
        """Fire the registered callback or fall back to a console print."""
        if self._blanket_callback:
            self._blanket_callback(count, names)
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[BLANKET] {ts} lower-body occlusion detected. "
            f"{count}/{len(self.cfg.lower_body_indices)} invisible: "
            f"{', '.join(names)}"
        )

    @staticmethod
    def _make_result(
        flag: bool,
        invisible: int,
        names: List[str],
        consecutive: int,
        newly_raised: bool,
    ) -> Dict:
        """Build the standardised blanket result dict returned by ``update()``."""
        return {
            "blanket_flag": flag,
            "invisible_count": invisible,
            "invisible_names": names,
            "consecutive": consecutive,
            "newly_raised": newly_raised,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sleep detection  (ported from baby_monitor_ncnn.py)
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """YuNet face detector (OpenCV ≥ 4.8) with Haar cascade fallback."""

    def __init__(self, w: int, h: int) -> None:
        """Attempt to initialise the YuNet detector for frames of size *w* × *h*.

        Falls back silently to OpenCV's built-in Haar cascade if YuNet is
        unavailable (older OpenCV builds).
        """
        self._yunet = False
        try:
            self._det = cv2.FaceDetectorYN.create(
                model="",
                config="",
                input_size=(w, h),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=1,
            )
            self._yunet = True
            print("[FaceDetector] YuNet (OpenCV built-in)")
        except Exception:
            xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._det = cv2.CascadeClassifier(xml)
            print("[FaceDetector] Haar cascade fallback")

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Returns (x, y, w, h) of the best face, or None."""
        if self._yunet:
            _, faces = self._det.detect(frame)
            if faces is None or len(faces) == 0:
                return None
            f = faces[0]
            return (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._det.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) == 0:
                return None
            return tuple(sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0])


class FaceLandmarker:
    """PFLD-lite 68-point facial landmarks via ncnn."""

    PFLD_INPUT_SIZE = 112

    def __init__(self, param_path: str, bin_path: str, num_threads: int = 2) -> None:
        """Load PFLD-lite weights from *param_path* / *bin_path* via the ncnn runtime.

        Raises:
            RuntimeError: if the ``ncnn`` Python bindings are not installed.
        """
        if ncnn is None:
            raise RuntimeError(
                "ncnn Python bindings not installed. "
                "Run: pip install ncnn"
            )
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = num_threads
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        self._input_name = "input"
        self._output_name = "output"
        print("[FaceLandmarker] PFLD-lite loaded via ncnn")

    def get_landmarks(self, face_crop: np.ndarray) -> np.ndarray:
        """Returns (N, 2) array of (x, y) pixel coords in face_crop space."""
        h, w = face_crop.shape[:2]
        mat_in = ncnn.Mat.from_pixels_resize(
            face_crop,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            w, h,
            self.PFLD_INPUT_SIZE, self.PFLD_INPUT_SIZE,
        )
        mean_vals = [0.0, 0.0, 0.0]
        norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        ex = self.net.create_extractor()
        ex.input(self._input_name, mat_in)
        _, mat_out = ex.extract(self._output_name)

        pts = np.array(mat_out).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        return pts  # shape (98, 2) for PFLD-98


class SleepDetector:
    """
    Eye-state monitor using face detection + PFLD eye landmarks.

    Runs on the raw frame (not the resized pose input) for accuracy.
    Lazy-initializes FaceDetector and FaceLandmarker on the first frame
    so frame dimensions are known.
    """

    LEFT_EYE_IDX  = [60, 61, 63, 64, 65, 67]
    RIGHT_EYE_IDX = [68, 69, 71, 72, 73, 75]

    def __init__(self, config: SleepDetectionConfig) -> None:
        """Store config and reset the sleep state machine to "awake"."""
        self.cfg = config
        self._face_det: Optional[FaceDetector] = None
        self._landmarker: Optional[FaceLandmarker] = None
        self._initialized = False
        self._init_failed = False

        # State machine
        self._baby_state = "awake"
        self._closed_start_time: Optional[float] = None
        self._open_confirm_start: Optional[float] = None

    def load(self) -> bool:
        """Download PFLD model files if absent. Detector init is lazy on first frame."""
        if not self.cfg.enabled:
            return True
        try:
            download_pfld_models(self.cfg.models_dir)
            return True
        except Exception as exc:
            print(f"[SLEEP] Model download failed: {exc}")
            return False

    def _lazy_init(self, frame: np.ndarray) -> None:
        """Initialise ``FaceDetector`` and ``FaceLandmarker`` on the first frame.

        Must be called lazily (not at construction) because the frame dimensions
        needed by ``FaceDetector(w, h)`` are only available at runtime. Sets
        ``_init_failed = True`` permanently if ncnn is missing or model files
        cannot be loaded, causing subsequent ``update()`` calls to short-circuit.
        """
        if self._initialized or self._init_failed:
            return
        h, w = frame.shape[:2]
        try:
            self._face_det = FaceDetector(w, h)
            param_path = os.path.join(self.cfg.models_dir, "pfld.param")
            bin_path   = os.path.join(self.cfg.models_dir, "pfld.bin")
            self._landmarker = FaceLandmarker(
                param_path, bin_path, self.cfg.ncnn_threads
            )
            self._initialized = True
        except Exception as exc:
            print(f"[SLEEP] Init failed (sleep detection disabled): {exc}")
            self._init_failed = True

    def update(self, frame: np.ndarray) -> Dict:
        """Run one eye-state inference cycle on *frame* and advance the state machine.

        Steps:
        1. Lazy-init detectors on first call.
        2. Detect the largest face in *frame*.
        3. Crop + pad the face region and run PFLD landmark inference.
        4. Compute the average Eye Aspect Ratio (EAR) for both eyes.
        5. Advance the awake ↔ asleep state machine.

        Returns:
            Dict with keys: ``face_found``, ``ear``, ``eyes_closed``,
            ``baby_state`` (``"awake"`` or ``"asleep"``),
            ``newly_asleep``, ``newly_awake``.
        """
        if not self.cfg.enabled:
            return self._empty_sleep_result()

        self._lazy_init(frame)
        if not self._initialized:
            return self._empty_sleep_result()

        eyes_closed = False
        ear: Optional[float] = None
        face_found = False

        face = self._face_det.detect(frame)
        if face is not None:
            face_found = True
            x, y, fw, fh = face
            pad = int(0.1 * min(fw, fh))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + fw + pad)
            y2 = min(frame.shape[0], y + fh + pad)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                try:
                    landmarks = self._landmarker.get_landmarks(face_crop)
                    left_eye  = landmarks[self.LEFT_EYE_IDX]
                    right_eye = landmarks[self.RIGHT_EYE_IDX]
                    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                    eyes_closed = ear < self.cfg.ear_closed_threshold
                except Exception as exc:
                    print(f"[SLEEP] Landmark inference failed: {exc}")

        # State machine
        t = time.time()
        newly_asleep = False
        newly_awake  = False

        if eyes_closed:
            self._open_confirm_start = None
            if self._closed_start_time is None:
                self._closed_start_time = t
            if (
                self._baby_state == "awake"
                and (t - self._closed_start_time) >= self.cfg.closed_seconds_threshold
            ):
                self._baby_state = "asleep"
                newly_asleep = True
        else:
            self._closed_start_time = None
            if self._baby_state == "asleep":
                if self._open_confirm_start is None:
                    self._open_confirm_start = t
                if (t - self._open_confirm_start) >= self.cfg.open_confirm_seconds:
                    self._baby_state = "awake"
                    self._open_confirm_start = None
                    newly_awake = True

        return {
            "face_found": face_found,
            "ear": ear,
            "eyes_closed": eyes_closed,
            "baby_state": self._baby_state,
            "newly_asleep": newly_asleep,
            "newly_awake": newly_awake,
        }

    @staticmethod
    def _empty_sleep_result() -> Dict:
        """Return a zeroed-out sleep result used when detection is disabled or fails."""
        return {
            "face_found": False,
            "ear": None,
            "eyes_closed": False,
            "baby_state": "awake",
            "newly_asleep": False,
            "newly_awake": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Monitor logger  (append-only JSON-lines)
# ─────────────────────────────────────────────────────────────────────────────

class MonitorLogger:
    """
    Appends one JSON object per line to a .log file.

    Each entry always contains a "timestamp" (UTC ISO-8601) and "type" field.
    """

    def __init__(self, log_path: Path, status_interval_seconds: float) -> None:
        """Create the log directory if needed and configure the status interval.

        Args:
            log_path: Full path to the ``.log`` file (created or appended on write).
            status_interval_seconds: Minimum seconds between ``status`` log entries.
        """
        self._path = log_path
        self._interval = status_interval_seconds
        self._last_status_time: float = 0.0
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, entry: Dict) -> None:
        """Stamp *entry* with a UTC timestamp and append it as a JSON line."""
        entry["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def log_pose_risk(self, nose_conf: Optional[float], consecutive_count: int) -> None:
        """Log a confirmed risky-posture event with the triggering nose confidence."""
        self._write({
            "type": "pose_risk",
            "nose_confidence": round(nose_conf, 4) if nose_conf is not None else None,
            "consecutive_low_conf": consecutive_count,
        })

    def log_burst_start(self, new_fps: int) -> None:
        """Log the moment burst mode is activated, recording the new FPS."""
        self._write({"type": "burst_start", "new_fps": new_fps})

    def log_false_alarm(self, duration_seconds: Optional[float], reason: str) -> None:
        """Log a burst-mode false alarm.

        Args:
            duration_seconds: How long the burst lasted before being dismissed.
            reason: ``"nose_reappeared"`` or ``"timeout"``.
        """
        self._write({
            "type": "false_alarm",
            "reason": reason,
            "burst_duration_seconds": (
                round(duration_seconds, 2) if duration_seconds is not None else None
            ),
        })

    def log_blanket(self, invisible_count: int, invisible_names: List[str]) -> None:
        """Log a blanket-occlusion alert with the number and names of hidden keypoints."""
        self._write({
            "type": "blanket",
            "invisible_count": invisible_count,
            "invisible_names": invisible_names,
        })

    def log_sleep_state(self, new_state: str, ear: Optional[float]) -> None:
        """Log a sleep-state transition (``"asleep"`` or ``"awake"``) with the EAR value."""
        self._write({
            "type": "sleep_state",
            "new_state": new_state,
            "ear": round(ear, 4) if ear is not None else None,
        })

    def log_kp_divergence(self, divergences: List[Dict]) -> None:
        """Log keypoints whose confidence dropped significantly below their rolling mean."""
        self._write({"type": "kp_divergence", "keypoints": divergences})

    def log_status(
        self,
        nose_conf: Optional[float],
        baby_state: str,
        blanket_flag: bool,
        ear: Optional[float],
    ) -> bool:
        """Write a regular status entry. Returns True if an entry was written."""
        now = time.time()
        if now - self._last_status_time < self._interval:
            return False
        self._last_status_time = now
        self._write({
            "type": "status",
            "nose_confidence": round(nose_conf, 4) if nose_conf is not None else None,
            "baby_state": baby_state,
            "blanket_flag": blanket_flag,
            "ear": round(ear, 4) if ear is not None else None,
        })
        return True


# ─────────────────────────────────────────────────────────────────────────────
# NCNN pose inference + risk detection
# ─────────────────────────────────────────────────────────────────────────────

class NCNNPoseInference:
    """YOLO-NCNN pose inference engine with integrated risk detection and burst mode.

    Responsibilities:
    - Load an Ultralytics YOLO model from an NCNN export directory.
    - Run per-frame inference and extract keypoint confidences.
    - Maintain the burst-mode state machine: low nose confidence → burst FPS →
      confirmed risk or false alarm → return to normal FPS.
    - Track per-keypoint rolling confidence histories for divergence detection.
    - Save risky frames and periodic snapshots to ``output_dir``.

    Burst-mode sentinel pattern:
        After each call to ``check_risk_condition()`` or ``check_burst_timeout()``,
        ``process_single_frame()`` reads the boolean/value sentinels
        (``_just_activated_burst``, ``_just_deactivated_burst``,
        ``_last_false_alarm_duration``, ``_last_false_alarm_reason``) and resets
        them to their defaults.  This avoids injecting the logger into this class.
    """

    NOSE_INDEX = 0

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise counters, rolling histories, burst-mode sentinels, and output directory."""
        self.cfg = config
        self.risk = config.risk_detection
        self.model = None

        self.consecutive_low_conf = 0
        self.monitoring_start = time.time()
        self.monitoring_frame_count = 0
        self.total_frames_processed = 0
        self.last_save_time = time.time()
        self.is_currently_risky = False

        # Per-keypoint rolling confidence histories (17 COCO keypoints)
        self._kp_histories: Dict[int, deque] = {
            i: deque(maxlen=self.risk.confidence_history_size) for i in range(17)
        }

        # Burst-mode state
        self._burst_active = False
        self._burst_start_time = 0.0
        self._frame_processor: Optional[FrameProcessor] = None

        # Sentinels read (and reset) by process_single_frame after each call
        self._just_activated_burst = False
        self._just_deactivated_burst = False
        self._last_false_alarm_duration: Optional[float] = None
        self._last_false_alarm_reason: str = "nose_reappeared"

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._warning_cb: Optional[Callable[[np.ndarray, Optional[float]], None]] = None
        self._save_frame_cb: Optional[Callable[[np.ndarray, Optional[float]], None]] = None
        self._periodic_save_cb: Optional[Callable[[np.ndarray], None]] = None

    def load_model(self) -> bool:
        """Load the YOLO-NCNN pose model from ``config.ncnn_model_dir``.

        Ultralytics requires the directory name to end in ``_ncnn_model``.  If
        the provided path does not match this convention, the method first tries
        to create a compatibility symlink and falls back to ``shutil.copytree``.

        Returns:
            True on success, False on any error (message is printed).
        """
        try:
            from ultralytics import YOLO

            model_dir = Path(self.cfg.ncnn_model_dir).expanduser().resolve()
            if not model_dir.is_dir():
                print(f"[ERROR] NCNN model directory not found: {model_dir}")
                return False

            expected_bin   = model_dir / "model.ncnn.bin"
            expected_param = model_dir / "model.ncnn.param"
            if not (expected_bin.exists() and expected_param.exists()):
                print(
                    f"[WARNING] Expected NCNN files not found in {model_dir}. "
                    "Ultralytics may still resolve the backend."
                )

            load_path = model_dir
            if not model_dir.name.endswith("_ncnn_model"):
                compat_dir = model_dir.parent / f"{model_dir.name}_ncnn_model"
                if not compat_dir.exists():
                    try:
                        os.symlink(str(model_dir), str(compat_dir))
                        print(
                            "[INFO] Created Ultralytics compatibility symlink: "
                            f"{compat_dir}"
                        )
                    except OSError:
                        shutil.copytree(str(model_dir), str(compat_dir))
                        print(
                            "[INFO] Created Ultralytics compatibility copy: "
                            f"{compat_dir}"
                        )
                load_path = compat_dir

            self.model = YOLO(str(load_path), task="pose")
            print(f"[INFO] NCNN model loaded from: {load_path}")
            return True
        except Exception as exc:
            print(f"[ERROR] Failed to load NCNN model: {exc}")
            return False

    def run_inference(self, frame: np.ndarray):
        """Run YOLO inference on a single pre-processed *frame* and return raw results.

        Raises:
            RuntimeError: if ``load_model()`` has not been called successfully.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model(frame, verbose=False, imgsz=self.cfg.imgsz)

    def get_nose_confidence(self, results) -> Optional[float]:
        """Extract the nose keypoint confidence from YOLO results, or None on failure."""
        return self._get_keypoint_confidence(results, self.NOSE_INDEX)

    def get_all_keypoint_confidences(self, results) -> Optional[np.ndarray]:
        """Return a (17,) float32 array of all keypoint confidences, or None on failure."""
        try:
            if results and len(results) > 0:
                keypoints = results[0].keypoints
                if (
                    keypoints is not None
                    and len(keypoints) > 0
                    and keypoints.conf is not None
                ):
                    return keypoints.conf[0].cpu().numpy()
        except Exception as exc:
            print(f"[WARNING] Failed to read keypoint confidences: {exc}")
        return None

    def _get_keypoint_confidence(self, results, index: int) -> Optional[float]:
        """Extract confidence for keypoint at *index* from YOLO results, or None."""
        try:
            if results and len(results) > 0:
                keypoints = results[0].keypoints
                if (
                    keypoints is not None
                    and len(keypoints) > 0
                    and keypoints.conf is not None
                    and len(keypoints.conf) > 0
                ):
                    return float(keypoints.conf[0][index])
        except Exception as exc:
            print(f"[WARNING] Failed to read keypoint {index}: {exc}")
        return None

    def check_burst_timeout(self) -> bool:
        """
        If burst mode has been active for longer than burst_timeout_seconds
        without confirming a risk, declare a false alarm and return to normal FPS.
        Returns True if a timeout false alarm was triggered.
        """
        if not self._burst_active or self.is_currently_risky:
            return False
        now = time.time()
        if now - self._burst_start_time >= self.risk.burst_timeout_seconds:
            self._last_false_alarm_duration = now - self._burst_start_time
            self._last_false_alarm_reason = "timeout"
            self._burst_active = False
            self._just_deactivated_burst = True
            self.consecutive_low_conf = 0
            if self._frame_processor is not None:
                self._frame_processor.set_target_fps(self.risk.normal_fps)
            return True
        return False

    def check_risk_condition(
        self,
        nose_conf: Optional[float],
        frame: np.ndarray,
    ) -> bool:
        """
        Burst-aware risk detection:

        - Low confidence → activate burst FPS (once), increment counter.
        - Counter fills within burst window → confirmed risk → alert + return to normal FPS.
        - Confidence recovers before counter fills → false alarm → return to normal FPS.
        - Burst timeout handled separately in check_burst_timeout().
        """
        cfg = self.risk
        now = time.time()

        if nose_conf is None or nose_conf < cfg.risk_threshold:
            # Activate burst on first low-confidence frame
            if not self._burst_active:
                self._burst_active = True
                self._burst_start_time = now
                self._just_activated_burst = True
                new_fps = cfg.normal_fps * cfg.burst_fps_multiplier
                if self._frame_processor is not None:
                    if self.cfg.source_type == "endpoint":
                        # Snapshot sources have a fixed poll interval; warn the operator.
                        # For MJPEG sources burst is fully effective.
                        pass
                    self._frame_processor.set_target_fps(new_fps)

            self.consecutive_low_conf += 1

            if self.consecutive_low_conf >= cfg.consecutive_risk_frames:
                if not self.is_currently_risky:
                    self.is_currently_risky = True
                    self._emit_warning(frame, nose_conf)
                    self._save_risky_frame(frame, nose_conf)
                # Confirmed risk → return to normal FPS immediately
                self._burst_active = False
                self._just_deactivated_burst = True
                self._last_false_alarm_reason = "confirmed_risk"
                self._last_false_alarm_duration = None
                self.consecutive_low_conf = 0
                if self._frame_processor is not None:
                    self._frame_processor.set_target_fps(cfg.normal_fps)
                return True

        else:
            # Nose reappeared
            if self._burst_active:
                self._last_false_alarm_duration = now - self._burst_start_time
                self._last_false_alarm_reason = "nose_reappeared"
                self._burst_active = False
                self._just_deactivated_burst = True
                if self._frame_processor is not None:
                    self._frame_processor.set_target_fps(cfg.normal_fps)

            self.consecutive_low_conf = 0
            self.is_currently_risky = False

        return False

    def check_keypoint_divergence(
        self, kp_confs: Optional[np.ndarray]
    ) -> List[Dict]:
        """
        Compare each of the 17 keypoint confidences against its rolling mean.
        Returns a list of keypoints whose confidence has dropped significantly.
        If kp_confs is None (no detection), histories are NOT updated to avoid
        poisoning the rolling means with zeros.
        """
        if kp_confs is None:
            return []
        divergences: List[Dict] = []
        margin = self.risk.confidence_drop_margin
        for i in range(17):
            hist = self._kp_histories[i]
            current = float(kp_confs[i]) if i < len(kp_confs) else 0.0
            if len(hist) >= self.risk.confidence_history_size:
                mean = float(np.mean(list(hist)))
                if current < mean - margin:
                    divergences.append({
                        "index": i,
                        "name": COCO_KP_NAMES[i],
                        "current": round(current, 4),
                        "mean": round(mean, 4),
                        "delta": round(current - mean, 4),
                    })
            hist.append(current)
        return divergences

    def check_monitoring_condition(self) -> bool:
        """Return True (and reset the counter/timer) when a monitoring checkpoint is due.

        Uses time-based intervals when ``priority_index == 0``, or frame-count
        intervals when ``priority_index == 1``.
        """
        now = time.time()
        elapsed = now - self.monitoring_start

        if self.risk.priority_index == 0:
            triggered = elapsed >= self.risk.monitoring_seconds
        else:
            triggered = self.monitoring_frame_count >= self.risk.monitoring_frames

        if triggered:
            self.monitoring_start = now
            self.monitoring_frame_count = 0
            return True
        return False

    def check_periodic_save(self, frame: np.ndarray) -> bool:
        """Save *frame* as a periodic snapshot if the save interval has elapsed.

        Skips the save when the pipeline is already in a confirmed risky state
        (risky frames are saved separately by ``_save_risky_frame``).
        Returns True if a save was triggered.
        """
        if self.is_currently_risky:
            return False
        now = time.time()
        if now - self.last_save_time >= self.risk.save_interval_seconds:
            self.last_save_time = now
            self._emit_periodic_save(frame)
            return True
        return False

    def get_annotated_frame(self, results) -> Optional[np.ndarray]:
        """Return a BGR frame with YOLO pose annotations drawn on it, or None on failure.

        This utility is available for external ``on_result`` callbacks that want
        to display or save the annotated output; it is not called internally.
        """
        try:
            if results and len(results) > 0:
                return results[0].plot()
        except Exception as exc:
            print(f"[WARNING] Failed to plot annotations: {exc}")
        return None

    def get_statistics(self) -> Dict:
        """Return a snapshot of key runtime counters for pipeline summary reporting."""
        return {
            "total_frames_processed": self.total_frames_processed,
            "consecutive_low_confidence": self.consecutive_low_conf,
            "is_currently_risky": self.is_currently_risky,
            "burst_active": self._burst_active,
            "monitoring_frame_count": self.monitoring_frame_count,
        }

    def set_warning_callback(
        self,
        cb: Callable[[np.ndarray, Optional[float]], None],
    ) -> None:
        """Register ``cb(frame, nose_conf)`` to replace the default console risk warning."""
        self._warning_cb = cb

    def set_save_frame_callback(
        self,
        cb: Callable[[np.ndarray, Optional[float]], None],
    ) -> None:
        """Register ``cb(frame, nose_conf)`` to replace default risky-frame disk save."""
        self._save_frame_cb = cb

    def set_periodic_save_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        """Register ``cb(frame)`` to replace the default periodic snapshot disk save."""
        self._periodic_save_cb = cb

    def _saveable_frame(self, frame: np.ndarray) -> np.ndarray:
        """Ensure *frame* is in BGR order for ``cv2.imwrite``."""
        return _frame_to_bgr(frame, self.cfg.frame_processing.color_order)

    def _emit_warning(self, frame: np.ndarray, conf: Optional[float]) -> None:
        """Fire the warning callback or print a console message on confirmed risk."""
        if self._warning_cb:
            self._warning_cb(frame, conf)
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[WARNING] {ts} risky posture detected (nose conf={conf})")

    def _save_risky_frame(self, frame: np.ndarray, conf: Optional[float]) -> None:
        """Save *frame* to disk (or delegate to callback) when a risk is confirmed."""
        if self._save_frame_cb:
            self._save_frame_cb(frame, conf)
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"risky_{ts}.jpg"
        cv2.imwrite(str(path), self._saveable_frame(frame))
        print(f"[INFO] Risky frame saved -> {path}")

    def _emit_periodic_save(self, frame: np.ndarray) -> None:
        """Save a periodic snapshot to disk (or delegate to callback)."""
        if self._periodic_save_cb:
            self._periodic_save_cb(frame)
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"periodic_{ts}.jpg"
        cv2.imwrite(str(path), self._saveable_frame(frame))
        print(f"[INFO] Periodic frame saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark reporter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdvancedBenchmarkReporter:
    """Accumulates per-frame timing and event statistics and writes benchmark reports.

    Timing data (preprocess / inference / postprocess) is read directly from
    Ultralytics' ``result.speed`` dict.  Reports can be saved as JSON
    (machine-readable) or Markdown (human-readable).

    Usage::

        reporter = AdvancedBenchmarkReporter()
        reporter.start()
        # ... process frames ...
        reporter.record_frame(result, elapsed_ms)
        reporter.stop()
        reporter.write_json("benchmark.json")
        reporter.write_markdown("benchmark.md")
    """

    total_frames_read: int = 0
    frames_processed: int = 0
    frames_skipped: int = 0
    risk_triggers: int = 0
    periodic_saves: int = 0
    blanket_flags_raised: int = 0

    frame_times_ms: List[float] = field(default_factory=list)
    preprocess_ms: List[float] = field(default_factory=list)
    inference_ms: List[float] = field(default_factory=list)
    postprocess_ms: List[float] = field(default_factory=list)

    blanket_flag_events: List[Dict] = field(default_factory=list)

    _start_time: float = 0.0
    _end_time: float = 0.0
    _env_meta: Dict[str, str] = field(default_factory=dict)

    model_format: str = "ncnn"
    source_type: str = "endpoint"

    def start(self) -> None:
        """Record the session start time and capture environment metadata."""
        self._start_time = time.time()
        self._env_meta = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cv2_version": cv2.__version__,
            "numpy_version": np.__version__,
        }

    def stop(self) -> None:
        """Record the session end time (used to compute wall-clock duration)."""
        self._end_time = time.time()

    def record_frame(self, result: Dict, elapsed_ms: float) -> None:
        """Accumulate stats for one frame.

        Args:
            result: The dict returned by ``process_single_frame()``.
            elapsed_ms: End-to-end wall time for that frame in milliseconds.
        """
        self.total_frames_read += 1

        if not result.get("processed"):
            self.frames_skipped += 1
            return

        self.frames_processed += 1
        self.frame_times_ms.append(elapsed_ms)

        yolo_results = result.get("results")
        if yolo_results and len(yolo_results) > 0:
            speed = getattr(yolo_results[0], "speed", None)
            if speed and isinstance(speed, dict):
                if "preprocess" in speed:
                    self.preprocess_ms.append(speed["preprocess"])
                if "inference" in speed:
                    self.inference_ms.append(speed["inference"])
                if "postprocess" in speed:
                    self.postprocess_ms.append(speed["postprocess"])

        if result.get("is_risky"):
            self.risk_triggers += 1
        if result.get("periodic_saved"):
            self.periodic_saves += 1

        blanket = result.get("blanket", {})
        if blanket.get("newly_raised"):
            self.blanket_flags_raised += 1
            self.blanket_flag_events.append(
                {
                    "frame_index": self.frames_processed,
                    "invisible_count": blanket.get("invisible_count"),
                    "invisible_names": blanket.get("invisible_names"),
                    "consecutive": blanket.get("consecutive"),
                }
            )

    def write_json(self, path: str) -> None:
        """Serialise the full benchmark report to a JSON file at *path*."""
        report = self._build_report()
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"[INFO] Benchmark JSON saved -> {output_path}")

    def write_markdown(self, path: str) -> None:
        """Write a human-readable Markdown benchmark report to *path*."""
        report = self._build_report()
        env     = report["environment"]
        session = report["session"]
        timing  = report["timing_ms"]
        events  = report["events"]
        blanket = report["blanket_detection"]

        lines = [
            "# Standalone Pi NCNN Benchmark Report",
            "",
            "## Environment",
            "",
            "| Key | Value |",
            "|-----|-------|",
        ]
        for key, value in env.items():
            lines.append(f"| {key} | {value} |")

        lines.extend(
            [
                "",
                "## Session",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Wall time (s) | {session['wall_seconds']} |",
                f"| Total frames read | {session['total_frames_read']} |",
                f"| Frames processed | {session['frames_processed']} |",
                f"| Frames skipped | {session['frames_skipped']} |",
                f"| Effective FPS | {session['effective_processed_fps']} |",
                "",
                "## Timing (ms)",
                "",
                "| Stage | Avg | Min | Max |",
                "|-------|-----|-----|-----|",
            ]
        )
        for stage in ("end_to_end", "preprocess", "inference", "postprocess"):
            stats = timing[stage]
            lines.append(
                f"| {stage} | {stats['avg']} | {stats['min']} | {stats['max']} |"
            )

        lines.extend(
            [
                "",
                "## Events",
                "",
                "| Event | Count |",
                "|-------|-------|",
                f"| Risk triggers | {events['risk_triggers']} |",
                f"| Periodic saves | {events['periodic_saves']} |",
                f"| Blanket flags | {events['blanket_flags_raised']} |",
            ]
        )

        if blanket["flag_events"]:
            lines.extend(
                [
                    "",
                    "## Blanket Events",
                    "",
                    "| Frame | Invisible Count | Names | Consecutive |",
                    "|-------|-----------------|-------|-------------|",
                ]
            )
            for event in blanket["flag_events"]:
                names = ", ".join(event.get("invisible_names", []))
                lines.append(
                    f"| {event['frame_index']} | {event['invisible_count']} "
                    f"| {names} | {event['consecutive']} |"
                )

        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"[INFO] Benchmark Markdown saved -> {output_path}")

    def _build_report(self) -> Dict:
        """Assemble the full report dict from accumulated stats."""
        wall = max(self._end_time - self._start_time, 1e-6)
        fps  = self.frames_processed / wall if wall > 0 else 0.0

        return {
            "environment": {
                **self._env_meta,
                "model_format": self.model_format,
                "source_type": self.source_type,
            },
            "session": {
                "wall_seconds": round(wall, 3),
                "total_frames_read": self.total_frames_read,
                "frames_processed": self.frames_processed,
                "frames_skipped": self.frames_skipped,
                "effective_processed_fps": round(fps, 2),
            },
            "timing_ms": {
                "end_to_end": self._stats(self.frame_times_ms),
                "preprocess":  self._stats(self.preprocess_ms),
                "inference":   self._stats(self.inference_ms),
                "postprocess": self._stats(self.postprocess_ms),
            },
            "events": {
                "risk_triggers":      self.risk_triggers,
                "periodic_saves":     self.periodic_saves,
                "blanket_flags_raised": self.blanket_flags_raised,
            },
            "blanket_detection": {
                "total_flags": self.blanket_flags_raised,
                "flag_events": self.blanket_flag_events,
            },
        }

    @staticmethod
    def _stats(values: List[float]) -> Dict[str, Optional[float]]:
        """Return ``{"avg", "min", "max"}`` for *values*, or all-None if empty."""
        if not values:
            return {"avg": None, "min": None, "max": None}
        return {
            "avg": round(sum(values) / len(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame processing
# ─────────────────────────────────────────────────────────────────────────────

def _empty_result() -> Dict:
    """Return a zeroed-out result dict used as the base for every frame's output.

    Keys:
        processed: False until the frame passes the FrameProcessor gate.
        nose_confidence: Float confidence of the nose keypoint, or None.
        is_risky: True when a confirmed risky posture was detected.
        monitoring_triggered: True on a monitoring checkpoint frame.
        periodic_saved: True when a periodic snapshot was saved.
        results: Raw Ultralytics YOLO results object (or None).
        kp_divergences: List of per-keypoint divergence dicts.
        burst_activated: True on the frame that started burst mode.
        burst_false_alarm: True on the frame that ended burst without confirming risk.
        burst_false_alarm_duration: Seconds spent in burst before dismissal.
        burst_false_alarm_reason: ``"nose_reappeared"`` or ``"timeout"``.
        blanket: Blanket detector status dict.
        sleep: Sleep detector status dict.
    """
    return {
        "processed": False,
        "nose_confidence": None,
        "is_risky": False,
        "monitoring_triggered": False,
        "periodic_saved": False,
        "results": None,
        "kp_divergences": [],
        "burst_activated": False,
        "burst_false_alarm": False,
        "burst_false_alarm_duration": None,
        "burst_false_alarm_reason": "nose_reappeared",
        "blanket": {
            "blanket_flag": False,
            "invisible_count": 0,
            "invisible_names": [],
            "consecutive": 0,
            "newly_raised": False,
        },
        "sleep": SleepDetector._empty_sleep_result(),
    }


def process_single_frame(
    raw_frame: np.ndarray,
    *,
    frame_processor: FrameProcessor,
    detector: NCNNPoseInference,
    blanket_detector: BlanketDetector,
    sleep_detector: SleepDetector,
) -> Dict:
    """Process one raw frame through the full pipeline and return a result dict.

    Steps (all keyword-only arguments are mandatory):
    1. Gate the frame through ``frame_processor``; skip if rate-limited.
    2. Check burst timeout before running inference.
    3. Run YOLO-NCNN pose inference on the pre-processed frame.
    4. Extract nose confidence and advance the risk / burst state machine.
    5. Read and reset burst-mode sentinels onto the result dict.
    6. Update blanket detector with all 17 keypoint confidences.
    7. Compute per-keypoint confidence divergences.
    8. Check monitoring and periodic-save conditions.
    9. Run sleep detection on the *raw* (non-resized) frame.

    Returns:
        A result dict with the structure described in ``_empty_result()``.
        ``result["processed"]`` is False when the frame was gated out.
    """
    result = _empty_result()

    should_process, prepared = frame_processor.prepare(raw_frame)
    if not should_process or prepared is None:
        return result

    result["processed"] = True
    detector.total_frames_processed += 1
    detector.monitoring_frame_count += 1

    # Check burst timeout before running inference
    detector.check_burst_timeout()

    results = detector.run_inference(prepared)
    result["results"] = results

    nose_conf = detector.get_nose_confidence(results)
    result["nose_confidence"] = nose_conf
    result["is_risky"] = detector.check_risk_condition(nose_conf, raw_frame)

    # Read and reset burst sentinels
    result["burst_activated"]             = detector._just_activated_burst
    result["burst_false_alarm"]           = detector._just_deactivated_burst
    result["burst_false_alarm_duration"]  = detector._last_false_alarm_duration
    result["burst_false_alarm_reason"]    = detector._last_false_alarm_reason
    detector._just_activated_burst        = False
    detector._just_deactivated_burst      = False
    detector._last_false_alarm_duration   = None

    kp_confs = detector.get_all_keypoint_confidences(results)
    result["blanket"]        = blanket_detector.update(kp_confs)
    result["kp_divergences"] = detector.check_keypoint_divergence(kp_confs)

    result["monitoring_triggered"] = detector.check_monitoring_condition()
    result["periodic_saved"]       = detector.check_periodic_save(raw_frame)

    # Sleep detection runs on the raw (non-resized) frame for accuracy
    result["sleep"] = sleep_detector.update(raw_frame)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline loop
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    frame_source: Iterator[Tuple[bool, Optional[np.ndarray]]],
    config: PipelineConfig,
    *,
    on_result: Optional[Callable[[Dict, np.ndarray], None]] = None,
) -> Dict:
    """Run the full monitoring pipeline until the source is exhausted or interrupted.

    Initialises all subsystems (frame processor, NCNN pose detector, blanket
    detector, sleep detector, logger, optional benchmark reporter), loads models,
    then loops over *frame_source* calling ``process_single_frame()`` for each
    frame.  All event logging and optional ``on_result`` callback dispatch happen
    here.

    Args:
        frame_source: An iterator yielding ``(ok, frame)`` pairs.  Iteration
            stops on ``(False, None)`` or when ``KeyboardInterrupt`` is raised.
        config: Full pipeline configuration.
        on_result: Optional callback invoked for each processed frame with
            ``(result_dict, raw_frame)``.

    Returns:
        A statistics dict (from ``detector.get_statistics()``) augmented with
        blanket counts, frame totals, and source type.
    """
    frame_processor  = FrameProcessor(config.frame_processing)
    detector         = NCNNPoseInference(config)
    detector._frame_processor = frame_processor          # wire burst mode
    blanket_detector = BlanketDetector(config.blanket_detection)
    sleep_detector   = SleepDetector(config.sleep_detection)

    log_path = (
        Path(config.log_file).expanduser()
        if config.log_file
        else Path(config.output_dir).expanduser() / "monitor.log"
    )
    logger = MonitorLogger(log_path, config.status_log_interval_seconds)

    reporter: Optional[AdvancedBenchmarkReporter] = None
    if config.benchmark or config.report_json or config.report_md:
        reporter = AdvancedBenchmarkReporter()
        reporter.model_format = "ncnn"
        reporter.source_type  = config.source_type

    print("[PIPELINE] Loading NCNN pose model...")
    if not detector.load_model():
        raise RuntimeError(f"Failed to load NCNN model from: {config.ncnn_model_dir}")

    print("[PIPELINE] Preparing sleep detection models...")
    sleep_detector.load()   # downloads PFLD if needed; lazy-inits on first frame

    print(
        "[PIPELINE] Starting "
        f"source_type={config.source_type} "
        f"normal_fps={config.risk_detection.normal_fps} "
        f"burst_fps={config.risk_detection.normal_fps * config.risk_detection.burst_fps_multiplier} "
        f"max_frames={config.frame_processing.max_frames} "
        f"blanket_detection={'ON' if config.blanket_detection.enabled else 'OFF'} "
        f"sleep_detection={'ON' if config.sleep_detection.enabled else 'OFF'}"
    )
    print(f"[PIPELINE] Log file: {log_path}")

    if reporter:
        reporter.start()

    loop_start   = time.time()
    frames_read  = 0

    try:
        for ok, raw_frame in frame_source:
            if config.max_seconds and (time.time() - loop_start) > config.max_seconds:
                print("[PIPELINE] Max seconds reached. Stopping.")
                break

            if not ok or raw_frame is None:
                print("[PIPELINE] Source ended. Stopping.")
                break

            frames_read += 1

            t0 = time.perf_counter()
            result = process_single_frame(
                raw_frame,
                frame_processor=frame_processor,
                detector=detector,
                blanket_detector=blanket_detector,
                sleep_detector=sleep_detector,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if reporter:
                reporter.record_frame(result, elapsed_ms)

            if result["processed"]:
                # ── Logging ───────────────────────────────────────────────
                if result["burst_activated"]:
                    new_fps = (
                        config.risk_detection.normal_fps
                        * config.risk_detection.burst_fps_multiplier
                    )
                    logger.log_burst_start(new_fps)

                if result["burst_false_alarm"]:
                    logger.log_false_alarm(
                        result["burst_false_alarm_duration"],
                        result["burst_false_alarm_reason"],
                    )

                if result["is_risky"]:
                    logger.log_pose_risk(
                        result["nose_confidence"],
                        detector.consecutive_low_conf,
                    )

                if result.get("blanket", {}).get("newly_raised"):
                    b = result["blanket"]
                    logger.log_blanket(b["invisible_count"], b["invisible_names"])

                sleep = result.get("sleep", {})
                if sleep.get("newly_asleep"):
                    logger.log_sleep_state("asleep", sleep.get("ear"))
                if sleep.get("newly_awake"):
                    logger.log_sleep_state("awake", sleep.get("ear"))

                if result["kp_divergences"]:
                    logger.log_kp_divergence(result["kp_divergences"])

                logger.log_status(
                    result["nose_confidence"],
                    sleep.get("baby_state", "unknown"),
                    result.get("blanket", {}).get("blanket_flag", False),
                    sleep.get("ear"),
                )
                # ── User callback ─────────────────────────────────────────
                if on_result:
                    on_result(result, raw_frame)

            if frame_processor.budget_exhausted():
                print("[PIPELINE] Frame budget exhausted. Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[PIPELINE] Interrupted by user.")

    finally:
        if reporter:
            reporter.stop()
            if config.report_json:
                reporter.write_json(config.report_json)
            elif config.benchmark:
                default_report = str(
                    Path(config.output_dir).expanduser() / "benchmark_report.json"
                )
                reporter.write_json(default_report)
            if config.report_md:
                reporter.write_markdown(config.report_md)

    stats = detector.get_statistics()
    stats["blanket_flags_raised"]    = blanket_detector.total_flags_raised
    stats["blanket_flag_history"]    = blanket_detector.flag_history
    stats["frames_read"]             = frames_read
    stats["frames_passed_to_model"]  = frame_processor.frames_processed
    stats["source_type"]             = config.source_type

    print("\n[PIPELINE] Final Statistics:")
    for key, value in stats.items():
        if key == "blanket_flag_history":
            print(f"       {key}: {len(value)} event(s)")
        else:
            print(f"       {key}: {value}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Frame sources
# ─────────────────────────────────────────────────────────────────────────────

class VideoCaptureIterator:
    """Wrap ``cv2.VideoCapture`` as a ``(ok, frame)`` iterator.

    Accepts a camera index (integer string), a local video file path, or any
    URL that OpenCV can open (e.g., RTSP streams).  Used as the fallback source
    when ``--source`` is provided instead of ``--endpoint-url``.
    """

    def __init__(self, source: str) -> None:
        """Convert a digit string to an integer camera index, keep other strings as-is."""
        self._source = int(source) if source.isdigit() else source
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the capture source. Returns False if the source cannot be opened."""
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            print(f"[ERROR] Cannot open source: {self._source}")
            return False
        print(f"[SOURCE] OpenCV source opened: {self._source}")
        return True

    def close(self) -> None:
        """Release the underlying ``cv2.VideoCapture`` handle."""
        if self._cap is not None:
            self._cap.release()
            print("[SOURCE] OpenCV source released.")

    def __iter__(self) -> Iterator[Tuple[bool, Optional[np.ndarray]]]:
        """Yield ``(ok, frame)`` until the capture source returns a failed read."""
        if self._cap is None:
            return
        while True:
            ok, frame = self._cap.read()
            yield ok, frame
            if not ok:
                return


class HttpSnapshotFrameSource:
    """HTTP frame source that polls a single-image endpoint (snapshot mode).

    Each iteration fires one GET request and decodes the response body as a
    JPEG/PNG image.  A minimum inter-request interval enforces the target FPS.

    Note:
        Burst mode (``FrameProcessor.set_target_fps``) does **not** speed up
        this source because ``_min_interval_seconds`` is fixed at construction
        time.  Burst is only fully effective with ``HttpMjpegFrameSource``.
    """

    def __init__(
        self,
        config: EndpointSourceConfig,
        *,
        min_interval_seconds: float,
    ) -> None:
        """Store connection config and the minimum poll interval."""
        self.cfg = config
        self._min_interval_seconds = max(min_interval_seconds, 0.0)
        self._session = None
        self._last_request_time = 0.0

    def open(self) -> bool:
        """Create a ``requests.Session``. Returns False if ``requests`` is not installed."""
        if requests is None:
            print("[ERROR] requests is not installed. Install it with: pip install requests")
            return False
        self._session = requests.Session()
        print(f"[SOURCE] Snapshot endpoint opened: {self.cfg.url}")
        return True

    def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None
            print("[SOURCE] Snapshot endpoint session closed.")

    def __iter__(self) -> Iterator[Tuple[bool, Optional[np.ndarray]]]:
        """Yield ``(True, frame)`` on success; ``(False, None)`` after too many errors."""
        if self._session is None:
            return

        errors = 0
        auth = None
        if self.cfg.auth_user is not None:
            auth = (self.cfg.auth_user, self.cfg.auth_password or "")

        while True:
            delay = self._min_interval_seconds - (
                time.monotonic() - self._last_request_time
            )
            if delay > 0:
                time.sleep(delay)

            try:
                self._last_request_time = time.monotonic()
                with self._session.get(
                    self.cfg.url,
                    headers=self.cfg.headers,
                    timeout=self.cfg.timeout_seconds,
                    verify=self.cfg.verify_ssl,
                    auth=auth,
                ) as response:
                    response.raise_for_status()
                    frame = _decode_image_bytes(response.content)
                if frame is None:
                    raise ValueError("Unable to decode endpoint image bytes.")

                errors = 0
                yield True, frame

            except Exception as exc:
                errors += 1
                print(f"[SOURCE] Snapshot fetch failed ({errors}): {exc}")
                if 0 < self.cfg.max_consecutive_errors <= errors:
                    yield False, None
                    return
                time.sleep(self.cfg.retry_delay_seconds)


class HttpMjpegFrameSource:
    """HTTP frame source that reads a continuous MJPEG multipart stream.

    Buffers incoming TCP chunks and scans for JPEG start (``\\xff\\xd8``) and
    end (``\\xff\\xd9``) markers to extract individual frames.  A
    ``min_interval_seconds`` gate drops frames that arrive faster than the
    target FPS, but burst mode can lower this gate at runtime by updating
    ``FrameProcessor._frame_interval`` (the source itself keeps streaming).
    """

    def __init__(
        self,
        config: EndpointSourceConfig,
        *,
        min_interval_seconds: float,
    ) -> None:
        """Store connection config and the initial minimum yield interval."""
        self.cfg = config
        self._min_interval_seconds = max(min_interval_seconds, 0.0)
        self._session = None
        self._last_yield_time = 0.0

    def open(self) -> bool:
        """Create a ``requests.Session``. Returns False if ``requests`` is not installed."""
        if requests is None:
            print("[ERROR] requests is not installed. Install it with: pip install requests")
            return False
        self._session = requests.Session()
        print(f"[SOURCE] MJPEG endpoint opened: {self.cfg.url}")
        return True

    def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None
            print("[SOURCE] MJPEG endpoint session closed.")

    def __iter__(self) -> Iterator[Tuple[bool, Optional[np.ndarray]]]:
        """Yield ``(True, frame)`` for each decoded JPEG; ``(False, None)`` after too many errors."""
        if self._session is None:
            return

        errors = 0
        auth = None
        if self.cfg.auth_user is not None:
            auth = (self.cfg.auth_user, self.cfg.auth_password or "")

        while True:
            try:
                with self._session.get(
                    self.cfg.url,
                    headers=self.cfg.headers,
                    timeout=self.cfg.timeout_seconds,
                    verify=self.cfg.verify_ssl,
                    auth=auth,
                    stream=True,
                ) as response:
                    response.raise_for_status()

                    buffer = b""
                    for chunk in response.iter_content(chunk_size=4096):
                        if not chunk:
                            continue
                        buffer += chunk

                        start = buffer.find(b"\xff\xd8")
                        end   = buffer.find(b"\xff\xd9")
                        while start != -1 and end != -1 and end > start:
                            jpg    = buffer[start : end + 2]
                            buffer = buffer[end + 2 :]

                            frame = _decode_image_bytes(jpg)
                            if frame is not None:
                                now = time.monotonic()
                                if (
                                    self._min_interval_seconds <= 0
                                    or now - self._last_yield_time >= self._min_interval_seconds
                                ):
                                    self._last_yield_time = now
                                    errors = 0
                                    yield True, frame

                            start = buffer.find(b"\xff\xd8")
                            end   = buffer.find(b"\xff\xd9")

                raise RuntimeError("MJPEG stream ended.")

            except Exception as exc:
                errors += 1
                print(f"[SOURCE] MJPEG stream failed ({errors}): {exc}")
                if 0 < self.cfg.max_consecutive_errors <= errors:
                    yield False, None
                    return
                time.sleep(self.cfg.retry_delay_seconds)


def detect_endpoint_mode(config: EndpointSourceConfig) -> str:
    """Determine whether the endpoint serves snapshots or an MJPEG stream.

    When ``config.mode`` is not ``"auto"`` the value is returned immediately.
    Otherwise, a HEAD-style streaming GET is issued and the ``Content-Type``
    header is inspected: ``"multipart/"`` or ``"mjpeg"`` → ``"mjpeg"``;
    anything else → ``"snapshot"``.
    """
    if config.mode != "auto":
        return config.mode
    if requests is None:
        raise RuntimeError("requests is required for endpoint mode detection.")

    auth = None
    if config.auth_user is not None:
        auth = (config.auth_user, config.auth_password or "")

    with requests.Session() as session:
        with session.get(
            config.url,
            headers=config.headers,
            timeout=config.timeout_seconds,
            verify=config.verify_ssl,
            auth=auth,
            stream=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()

    if "multipart/" in content_type or "mjpeg" in content_type:
        print(f"[SOURCE] Auto-detected MJPEG endpoint: {content_type}")
        return "mjpeg"

    print(f"[SOURCE] Auto-detected snapshot endpoint: {content_type or 'unknown'}")
    return "snapshot"


def build_frame_source(
    args: argparse.Namespace,
    pipeline_config: PipelineConfig,
) -> Tuple[Iterator[Tuple[bool, Optional[np.ndarray]]], Callable[[], None], str]:
    """Construct and open the appropriate frame source from CLI arguments.

    Returns:
        A 3-tuple: ``(iterator, close_fn, source_type_str)`` where
        *source_type_str* is ``"endpoint"`` or ``"opencv"``.

    Raises:
        RuntimeError: if the source cannot be opened or if neither
            ``--endpoint-url`` nor ``--source`` was provided.
    """
    if args.endpoint_url:
        endpoint_mode = detect_endpoint_mode(pipeline_config.endpoint_source)
        min_interval = (
            pipeline_config.endpoint_source.poll_interval_seconds
            if pipeline_config.endpoint_source.poll_interval_seconds is not None
            else 1.0 / max(pipeline_config.frame_processing.target_fps, 1)
        )

        if endpoint_mode == "mjpeg":
            source = HttpMjpegFrameSource(
                pipeline_config.endpoint_source,
                min_interval_seconds=min_interval,
            )
        else:
            source = HttpSnapshotFrameSource(
                pipeline_config.endpoint_source,
                min_interval_seconds=min_interval,
            )

        if not source.open():
            raise RuntimeError("Failed to open endpoint source.")
        return iter(source), source.close, "endpoint"

    if not args.source:
        raise RuntimeError("Either --endpoint-url or --source must be provided.")

    source = VideoCaptureIterator(args.source)
    if not source.open():
        raise RuntimeError(f"Failed to open source: {args.source}")
    return iter(source), source.close, "opencv"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser for the monitoring pipeline."""
    parser = argparse.ArgumentParser(
        prog="standalone_pi_pose_ncnn",
        description=(
            "Standalone NCNN infant monitoring pipeline for Raspberry Pi. "
            "Reads frames from an HTTP camera endpoint or an OpenCV source, "
            "runs YOLO-NCNN pose inference, and optionally sleep detection "
            "via PFLD eye landmarks. Events are written to a JSON-lines log file."
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the NCNN export directory (must contain model.ncnn.bin + model.ncnn.param).",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--endpoint-url",
        help="HTTP endpoint returning either snapshot images or an MJPEG stream.",
    )
    source_group.add_argument(
        "--source",
        help="Fallback OpenCV source: video path, URL, or integer camera index.",
    )

    parser.add_argument(
        "--endpoint-mode",
        choices=["auto", "snapshot", "mjpeg"],
        default="auto",
        help="Endpoint mode. 'auto' inspects the HTTP response content type.",
    )
    parser.add_argument(
        "--endpoint-timeout",
        type=float,
        default=5.0,
        help="Per-request timeout in seconds for endpoint fetches.",
    )
    parser.add_argument(
        "--endpoint-poll-interval",
        type=float,
        default=None,
        help="Snapshot polling interval in seconds. Default uses 1/target-fps.",
    )
    parser.add_argument(
        "--endpoint-retry-delay",
        type=float,
        default=0.5,
        help="Delay in seconds before retrying a failed endpoint fetch.",
    )
    parser.add_argument(
        "--endpoint-max-errors",
        type=int,
        default=10,
        help="Stop after this many consecutive endpoint errors. Use 0 for infinite retries.",
    )
    parser.add_argument(
        "--endpoint-header",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional HTTP header. Can be passed multiple times.",
    )
    parser.add_argument(
        "--endpoint-auth-user",
        default=None,
        help="Optional basic-auth username for the endpoint.",
    )
    parser.add_argument(
        "--endpoint-auth-password",
        default=None,
        help="Optional basic-auth password for the endpoint.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for HTTPS endpoints.",
    )

    # Frame processing
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument(
        "--target-fps",
        type=int,
        default=1,
        help="Normal processing rate (default 1 fps).",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--resize-enabled", action="store_true", default=True)
    parser.add_argument("--no-resize", dest="resize_enabled", action="store_false")
    parser.add_argument(
        "--resize-dim",
        type=int,
        nargs=2,
        default=[416, 416],
        metavar=("W", "H"),
    )
    parser.add_argument(
        "--resize-interp",
        type=str,
        default="LINEAR",
        choices=["NEAREST", "LINEAR", "AREA", "CUBIC", "LANCZOS4"],
    )
    parser.add_argument(
        "--color-order",
        choices=["RGB", "BGR"],
        default=None,
        help="Force incoming frame color order. Endpoint/OpenCV defaults to BGR.",
    )
    parser.add_argument(
        "--convert-to-bgr",
        action="store_true",
        default=None,
        help="Convert incoming RGB frames to BGR before inference.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize pixels to [0, 1] before inference.",
    )

    # Risk detection
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--consecutive-risk-frames", type=int, default=5)
    parser.add_argument("--confidence-history-size", type=int, default=10)
    parser.add_argument("--confidence-drop-margin", type=float, default=0.15)
    parser.add_argument("--monitoring-seconds", type=float, default=10.0)
    parser.add_argument("--monitoring-frames", type=int, default=50)
    parser.add_argument(
        "--monitoring-priority",
        type=int,
        choices=[0, 1],
        default=0,
        help="0 = seconds-based monitoring, 1 = frame-based monitoring.",
    )
    parser.add_argument("--save-interval-seconds", type=float, default=30.0)

    # Burst mode
    parser.add_argument(
        "--burst-multiplier",
        type=int,
        default=5,
        help="FPS multiplier during burst mode (default: 5× normal-fps).",
    )
    parser.add_argument(
        "--burst-timeout",
        type=float,
        default=10.0,
        help="Max seconds in burst mode before declaring a false alarm.",
    )
    parser.add_argument(
        "--normal-fps",
        type=int,
        default=None,
        help="Baseline FPS to return to after burst. Defaults to --target-fps.",
    )

    # Blanket detection
    parser.add_argument("--blanket-detection", action="store_true", default=True)
    parser.add_argument(
        "--no-blanket-detection",
        dest="blanket_detection",
        action="store_false",
    )
    parser.add_argument("--blanket-visibility-thresh", type=float, default=0.30)
    parser.add_argument("--blanket-min-invisible", type=int, default=4)
    parser.add_argument("--blanket-consec-frames", type=int, default=5)
    parser.add_argument("--blanket-cooldown", type=float, default=30.0)

    # Sleep detection
    parser.add_argument("--sleep-detection", action="store_true", default=True)
    parser.add_argument(
        "--no-sleep-detection",
        dest="sleep_detection",
        action="store_false",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.21,
        help="EAR value below which eyes are considered closed.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=10.0,
        help="Seconds eyes must be closed before declaring baby asleep.",
    )
    parser.add_argument(
        "--sleep-open-confirm",
        type=float,
        default=0.5,
        help="Seconds eyes must be open to confirm baby has woken.",
    )
    parser.add_argument(
        "--sleep-models-dir",
        type=str,
        default="./models",
        help="Directory to store/load PFLD model files.",
    )
    parser.add_argument("--sleep-ncnn-threads", type=int, default=2)

    # Output / logging
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./advanced_inference_output",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path for the JSON-lines log file. Default: {output-dir}/monitor.log",
    )
    parser.add_argument(
        "--status-log-interval",
        type=float,
        default=30.0,
        help="Seconds between regular status entries in the log.",
    )
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--report-json", type=str, default=None)
    parser.add_argument("--report-md", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", default=False)

    return parser


def args_to_config(args: argparse.Namespace) -> PipelineConfig:
    """Convert parsed CLI arguments to a fully-populated ``PipelineConfig``.

    Handles two special cases:
    - ``color_order``: defaults to ``"BGR"`` for endpoint/OpenCV sources.
    - ``normal_fps``: falls back to ``target_fps`` when ``--normal-fps`` is
      not explicitly set, so burst always returns to the correct baseline.
    """
    headers = _parse_header_values(args.endpoint_header)

    if args.color_order is not None:
        color_order = args.color_order
    elif args.endpoint_url or args.source:
        color_order = "BGR"
    else:
        color_order = "RGB"

    if args.convert_to_bgr is not None:
        convert_to_bgr = args.convert_to_bgr
    else:
        convert_to_bgr = color_order == "RGB"

    source_type = "endpoint" if args.endpoint_url else "opencv"
    normal_fps  = args.normal_fps if args.normal_fps is not None else args.target_fps

    return PipelineConfig(
        ncnn_model_dir=args.model,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
        source_type=source_type,
        max_seconds=args.max_seconds,
        benchmark=args.benchmark,
        report_json=args.report_json,
        report_md=args.report_md,
        status_log_interval_seconds=args.status_log_interval,
        log_file=args.log_file,
        frame_processing=FrameProcessingConfig(
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
            target_fps=args.target_fps,
            resize_enabled=args.resize_enabled,
            resize_dimensions=tuple(args.resize_dim),
            resize_interpolation=args.resize_interp,
            color_order=color_order,
            convert_to_bgr=convert_to_bgr,
            normalize=args.normalize,
        ),
        risk_detection=RiskDetectionConfig(
            risk_threshold=args.risk_threshold,
            consecutive_risk_frames=args.consecutive_risk_frames,
            confidence_history_size=args.confidence_history_size,
            confidence_drop_margin=args.confidence_drop_margin,
            monitoring_seconds=args.monitoring_seconds,
            monitoring_frames=args.monitoring_frames,
            priority_index=args.monitoring_priority,
            save_interval_seconds=args.save_interval_seconds,
            burst_fps_multiplier=args.burst_multiplier,
            burst_timeout_seconds=args.burst_timeout,
            normal_fps=normal_fps,
        ),
        blanket_detection=BlanketDetectionConfig(
            enabled=args.blanket_detection,
            visibility_threshold=args.blanket_visibility_thresh,
            min_invisible_keypoints=args.blanket_min_invisible,
            consecutive_frames_to_flag=args.blanket_consec_frames,
            cooldown_seconds=args.blanket_cooldown,
        ),
        sleep_detection=SleepDetectionConfig(
            enabled=args.sleep_detection,
            ear_closed_threshold=args.ear_threshold,
            closed_seconds_threshold=args.sleep_seconds,
            open_confirm_seconds=args.sleep_open_confirm,
            models_dir=args.sleep_models_dir,
            ncnn_threads=args.sleep_ncnn_threads,
        ),
        endpoint_source=EndpointSourceConfig(
            url=args.endpoint_url or "",
            mode=args.endpoint_mode,
            timeout_seconds=args.endpoint_timeout,
            poll_interval_seconds=args.endpoint_poll_interval,
            retry_delay_seconds=args.endpoint_retry_delay,
            max_consecutive_errors=args.endpoint_max_errors,
            verify_ssl=not args.insecure,
            headers=headers,
            auth_user=args.endpoint_auth_user,
            auth_password=args.endpoint_auth_password,
        ),
    )


def make_on_result_callback(
    *,
    quiet: bool,
) -> Callable[[Dict, np.ndarray], None]:
    """Return an ``on_result(result, frame)`` callback for console logging.

    When *quiet* is True the returned callback does nothing.  Otherwise it
    prints a one-line summary per processed frame showing nose confidence,
    risk/safe status, blanket flag, sleep state, and burst-mode markers.
    """
    def on_result(result: Dict, _: np.ndarray) -> None:
        if quiet:
            return
        nose   = result.get("nose_confidence")
        conf_text = f"{nose:.3f}" if nose is not None else "None"
        status = "RISKY" if result.get("is_risky") else "SAFE"
        blanket_text = " BLANKET!" if result.get("blanket", {}).get("blanket_flag") else ""
        sleep  = result.get("sleep", {})
        sleep_text = f" sleep={sleep['baby_state']}" if sleep.get("baby_state") else ""
        burst_text = " [BURST]" if result.get("burst_activated") else ""
        fa_text = " [FALSE-ALARM]" if result.get("burst_false_alarm") else ""
        print(
            f"[LOOP] nose_conf={conf_text} status={status}"
            f"{blanket_text}{sleep_text}{burst_text}{fa_text}"
        )

    return on_result


def main() -> None:
    """Entry point: parse CLI args, open the frame source, and run the pipeline."""
    parser = build_parser()
    args   = parser.parse_args()

    frame_source = None
    cleanup: Callable[[], None] = lambda: None

    try:
        config = args_to_config(args)
        frame_source, cleanup, source_type = build_frame_source(args, config)
        config.source_type = source_type

        on_result = make_on_result_callback(quiet=args.quiet)
        run_pipeline(frame_source, config, on_result=on_result)

    except KeyboardInterrupt:
        print("\n[MAIN] Stopped by user.")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
