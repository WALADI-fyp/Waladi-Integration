#!/usr/bin/env python3
"""
Waladi Baby Cry Detection — standalone single-file script
==========================================================
Uses YAMNet TFLite model with tflite-runtime.
No TensorFlow, no NCNN, no onnx2ncnn binary needed.

First run: downloads yamnet.tflite (~3 MB) and installs tflite-runtime.
Subsequent runs: starts immediately.

Usage:
    python3 cry_detection_standalone.py
"""

import os
import sys
import shutil
import subprocess
import tempfile
import time
import wave
import urllib.request
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  CONFIG — edit these to match your Pi setup
# ══════════════════════════════════════════════════════════════════════

# Where the .tflite model file will be saved
MODEL_DIR  = Path.home() / "waladi_models" / "yamnet"
MODEL_FILE = MODEL_DIR / "yamnet.tflite"

# INMP441 microphone ALSA device  (run `arecord -l` to confirm)
CARD_DEVICE   = "hw:2,0"
SRC_RATE      = 48000
SRC_FORMAT    = "S32_LE"
SRC_CHANNELS  = 2
OUT_RATE      = 16000
CHUNK_SECONDS = 1        # seconds of audio per inference call
GAIN_DB       = 7        # applied after normalization (keep ≤ 10)

# YAMNet class IDs for crying (standard YAMNet class map)
# 20 = Baby cry,  21 = Crying/sobbing,  23 = Whimper
CRY_CLASS_IDS = [20, 21, 23]

# Detection thresholds
CRY_PROB_THRESHOLD   = 0.35   # single-chunk score to count as "possible cry"
CRY_CONFIRM_SECONDS  = 3.0    # must stay above threshold this long → CRYING
SILENCE_CONFIRM_SECS = 2.0    # must stay below threshold this long → QUIET

# ══════════════════════════════════════════════════════════════════════
#  DEPENDENCY CHECK
# ══════════════════════════════════════════════════════════════════════

def _pip(*packages):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--break-system-packages", *packages],
        check=True,
    )


def ensure_sox():
    if shutil.which("sox") is None:
        print("[setup] Installing sox...")
        subprocess.run(["sudo", "apt-get", "install", "-y", "-q", "sox"], check=True)


def ensure_tflite_runtime():
    """Try tflite-runtime, fall back to ai-edge-litert (newer Pi OS)."""
    try:
        import tflite_runtime.interpreter  # noqa
        return
    except ImportError:
        pass
    try:
        import ai_edge_litert.interpreter  # noqa
        return
    except ImportError:
        pass

    print("[setup] Installing tflite-runtime...")
    try:
        _pip("tflite-runtime")
        import tflite_runtime.interpreter  # noqa
        return
    except Exception:
        pass

    print("[setup] tflite-runtime failed, trying ai-edge-litert...")
    _pip("ai-edge-litert")


def get_interpreter_class():
    """Return whichever Interpreter class is available."""
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        raise SystemExit(
            "[setup] Could not import tflite Interpreter. "
            "Try: pip install tflite-runtime --break-system-packages"
        )


# ══════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════════════

# Pre-built YAMNet TFLite model (~3.7 MB) — no conversion needed
MODEL_URLS = [
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/rpi/lite-model_yamnet_classification_tflite_1.tflite",
    "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite",
]


def _download(url: str, dest: Path) -> bool:
    try:
        print(f"[model] Trying: {url[:70]}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 100_000:   # sanity check — real model is ~3.7 MB
            return False
        dest.write_bytes(data)
        print(f"[model] Downloaded {len(data)/1e6:.1f} MB → {dest}")
        return True
    except Exception as e:
        print(f"[model]   Failed: {e}")
        return False


def ensure_model():
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 100_000:
        print(f"[model] Found existing model at {MODEL_FILE}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[model] Downloading YAMNet TFLite model (~3.7 MB)...")

    for url in MODEL_URLS:
        if _download(url, MODEL_FILE):
            return

    raise SystemExit(
        "\n[model] ERROR: Could not download the model automatically.\n"
        "  Download manually on another machine and copy to the Pi:\n\n"
        f"  scp yamnet.tflite pi@<PI_IP>:{MODEL_FILE}\n\n"
        "  Download URL:\n"
        f"  {MODEL_URLS[0]}\n"
    )


# ══════════════════════════════════════════════════════════════════════
#  AUDIO CAPTURE  (arecord → sox → 16-bit mono float32)
# ══════════════════════════════════════════════════════════════════════

def capture_chunk(tmp_dir: Path):
    import numpy as np

    raw_wav   = tmp_dir / "raw.wav"
    final_wav = tmp_dir / "final.wav"

    subprocess.run(
        [
            "arecord",
            "-D", CARD_DEVICE,
            "-c", str(SRC_CHANNELS),
            "-r", str(SRC_RATE),
            "-f", SRC_FORMAT,
            "-t", "wav",
            str(raw_wav),
            "-d", str(CHUNK_SECONDS),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    sox_cmd = [
        "sox", str(raw_wav),
        "-t", "wavpcm", "-b", "16", str(final_wav),
        "remix", "1",
        "gain", "-n",
    ]
    if GAIN_DB != 0:
        sox_cmd += ["gain", str(GAIN_DB)]
    sox_cmd += ["rate", str(OUT_RATE), "channels", "1"]

    subprocess.run(
        sox_cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    with wave.open(str(final_wav), "rb") as wf:
        raw = wf.readframes(wf.getnframes())

    return np.frombuffer(raw, dtype="int16").astype("float32") / 32768.0


# ══════════════════════════════════════════════════════════════════════
#  TFLITE INFERENCE
# ══════════════════════════════════════════════════════════════════════

def load_model():
    import numpy as np
    Interpreter = get_interpreter_class()

    interp = Interpreter(model_path=str(MODEL_FILE))
    interp.allocate_tensors()

    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    inp_shape = inp_details[0]["shape"]
    print(f"[model] Input shape: {inp_shape}, Output shape: {out_details[0]['shape']}")

    return interp, inp_details[0]["index"], out_details[0]["index"]


def run_inference(interp, inp_idx: int, out_idx: int, waveform) -> float:
    import numpy as np

    # YAMNet TFLite expects exactly 15600 samples (0.975s) or 16000 samples
    # Trim or pad to match what the model expects
    inp_shape    = interp.get_input_details()[0]["shape"]
    expected_len = int(inp_shape[0])
    n            = len(waveform)

    if n >= expected_len:
        w = waveform[:expected_len]
    else:
        w = np.pad(waveform, (0, expected_len - n))

    interp.set_tensor(inp_idx, w)
    interp.invoke()

    scores = interp.get_tensor(out_idx)   # shape: (num_classes,) or (N, num_classes)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    mean_scores = scores.mean(axis=0)
    return float(max(
        (mean_scores[i] for i in CRY_CLASS_IDS if i < len(mean_scores)),
        default=0.0,
    ))


# ══════════════════════════════════════════════════════════════════════
#  DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════

def _print_banner(state: str, cry_prob: float):
    if state == "crying":
        print(
            f"\n{'='*52}\n"
            f"  BABY CRYING DETECTED   (prob={cry_prob:.3f})\n"
            f"{'='*52}"
        )
    else:
        print(
            f"\n{'='*52}\n"
            f"  Cry stopped / quiet    (prob={cry_prob:.3f})\n"
            f"{'='*52}"
        )


def run_loop(interp, inp_idx: int, out_idx: int):
    state       = "no_cry"
    cry_start   = None
    nocry_start = None
    elapsed     = 0.0

    print(
        f"\n[cry] Listening on {CARD_DEVICE}  "
        f"({CHUNK_SECONDS}s chunks, threshold={CRY_PROB_THRESHOLD}) "
        f"— Ctrl+C to stop\n"
    )

    with tempfile.TemporaryDirectory(prefix="waladi_cry_") as tmp:
        tmp_dir = Path(tmp)

        while True:
            try:
                waveform = capture_chunk(tmp_dir)
            except subprocess.CalledProcessError as e:
                print(f"\n[cry] Audio capture failed: {e} — retrying in 2s")
                time.sleep(2)
                continue

            cry_prob = run_inference(interp, inp_idx, out_idx, waveform)
            now      = time.time()

            if cry_prob >= CRY_PROB_THRESHOLD:
                nocry_start = None
                if cry_start is None:
                    cry_start = now
                elapsed = now - cry_start
                if state == "no_cry" and elapsed >= CRY_CONFIRM_SECONDS:
                    state = "crying"
                    _print_banner(state, cry_prob)
            else:
                cry_start = None
                if nocry_start is None:
                    nocry_start = now
                elapsed = now - nocry_start
                if state == "crying" and elapsed >= SILENCE_CONFIRM_SECS:
                    state = "no_cry"
                    _print_banner(state, cry_prob)

            bar_len = 20
            filled  = int(min(cry_prob, 1.0) * bar_len)
            bar     = "#" * filled + "." * (bar_len - filled)
            lbl     = "CRYING" if state == "crying" else "quiet "
            print(
                f"\r[{bar}] prob={cry_prob:.3f}  [{lbl}]  "
                f"confirm={elapsed:.1f}s/{CRY_CONFIRM_SECONDS:.0f}s   ",
                end="",
                flush=True,
            )


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 52)
    print("  Waladi Cry Detection — startup")
    print("=" * 52)

    ensure_sox()
    ensure_tflite_runtime()
    ensure_model()

    print("[cry] Loading model...")
    interp, inp_idx, out_idx = load_model()
    print("[cry] Model ready — starting detection loop")

    try:
        run_loop(interp, inp_idx, out_idx)
    except KeyboardInterrupt:
        print("\n[cry] Stopped.")


if __name__ == "__main__":
    main()
