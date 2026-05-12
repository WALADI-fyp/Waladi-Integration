#!/usr/bin/env python3
"""
Waladi Baby Cry Detection — standalone single-file script
==========================================================
Continuous streaming audio with overlapping inference windows.
No gaps, no dropped cries.

Usage:
    python3 cry_detection_standalone.py
"""

import os
import sys
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

MODEL_DIR  = Path.home() / "waladi_models" / "yamnet"
MODEL_FILE = MODEL_DIR / "yamnet.tflite"

# INMP441 mic — run `arecord -l` to confirm card number
CARD_DEVICE  = "hw:2,0"
SRC_RATE     = 48000
SRC_CHANNELS = 2
OUT_RATE     = 16000

# Gain applied after normalization. Raise this if prob stays near 0.
GAIN_DB = 15

# Sliding window: model sees 15600 samples (~1s), we slide every HOP_SAMPLES
# Smaller hop = more frequent checks, catches cries faster
MODEL_SAMPLES = 15600          # fixed by YAMNet TFLite
HOP_SAMPLES   = 8000           # slide every 0.5s

# YAMNet class IDs for crying sounds
# 20=Baby cry  21=Crying/sobbing  22=Shout  23=Whimper  498=Child speech
CRY_CLASS_IDS = [20, 21, 23]

# Threshold — lower = more sensitive, higher = fewer false positives
# Start at 0.20 and raise if you get false positives
CRY_PROB_THRESHOLD   = 0.20
CRY_CONFIRM_SECONDS  = 2.0    # seconds above threshold before declared CRYING
SILENCE_CONFIRM_SECS = 2.0    # seconds below threshold before declared QUIET

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

def ensure_tflite():
    for mod in ["tflite_runtime.interpreter", "ai_edge_litert.interpreter"]:
        try:
            __import__(mod)
            return
        except ImportError:
            pass
    print("[setup] Installing tflite-runtime...")
    try:
        _pip("tflite-runtime")
        return
    except Exception:
        pass
    print("[setup] Falling back to ai-edge-litert...")
    _pip("ai-edge-litert")

def get_interpreter_class():
    for mod, cls in [
        ("tflite_runtime.interpreter", "Interpreter"),
        ("ai_edge_litert.interpreter", "Interpreter"),
    ]:
        try:
            m = __import__(mod, fromlist=[cls])
            return getattr(m, cls)
        except ImportError:
            pass
    raise SystemExit("[setup] No tflite Interpreter found. pip install tflite-runtime")

# ══════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════════════

MODEL_URLS = [
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/rpi/lite-model_yamnet_classification_tflite_1.tflite",
    "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite",
]

def ensure_model():
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 100_000:
        print(f"[model] Using existing model at {MODEL_FILE}")
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[model] Downloading YAMNet TFLite (~4 MB)...")
    for url in MODEL_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) < 100_000:
                continue
            MODEL_FILE.write_bytes(data)
            print(f"[model] Downloaded {len(data)/1e6:.1f} MB")
            return
        except Exception as e:
            print(f"[model] URL failed: {e}")
    raise SystemExit("[model] Could not download model. Copy yamnet.tflite manually to " + str(MODEL_FILE))

# ══════════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════

def load_model():
    Interpreter = get_interpreter_class()
    interp = Interpreter(model_path=str(MODEL_FILE))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"[model] Input: {inp['shape']}  Output: {out['shape']}")
    return interp, inp["index"], out["index"]

# ══════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════

def run_inference(interp, inp_idx, out_idx, waveform) -> float:
    import numpy as np
    n = len(waveform)
    if n >= MODEL_SAMPLES:
        w = waveform[:MODEL_SAMPLES]
    else:
        w = np.pad(waveform, (0, MODEL_SAMPLES - n))
    interp.set_tensor(inp_idx, w)
    interp.invoke()
    scores = interp.get_tensor(out_idx)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    mean_scores = scores.mean(axis=0)
    return float(max(
        (mean_scores[i] for i in CRY_CLASS_IDS if i < len(mean_scores)),
        default=0.0,
    ))

# ══════════════════════════════════════════════════════════════════════
#  CONTINUOUS STREAMING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_loop(interp, inp_idx, out_idx):
    import numpy as np

    # arecord → sox pipeline (continuous, no -d flag)
    # arecord outputs raw S32_LE stereo 48kHz
    # sox converts: left channel only, normalize, gain, resample to 16kHz mono 16-bit raw
    arecord_cmd = [
        "arecord",
        "-D", CARD_DEVICE,
        "-c", str(SRC_CHANNELS),
        "-r", str(SRC_RATE),
        "-f", "S32_LE",
        "-t", "raw",     # raw stream, no WAV header, runs forever
    ]
    # NOTE: gain -n (normalize) is NOT used — it requires reading the full
    # file to find the peak, so it buffers forever on an infinite stream.
    sox_cmd = [
        "stdbuf", "-o0",          # disable sox output buffering
        "sox",
        "--buffer", "1024",       # small input buffer = low latency
        "-t", "raw", "-b", "32", "-e", "signed-integer",
        "-r", str(SRC_RATE), "-c", str(SRC_CHANNELS), "-",   # stdin
        "-t", "raw", "-b", "16", "-e", "signed-integer",
        "-r", str(OUT_RATE), "-c", "1", "-",                  # stdout raw
        "remix", "1",             # left channel only
        "gain", str(GAIN_DB),     # fixed gain only, no -n normalize
    ]

    print(f"\n[cry] Starting continuous stream on {CARD_DEVICE}...")
    print(f"[cry] Threshold={CRY_PROB_THRESHOLD}  Confirm={CRY_CONFIRM_SECONDS}s  Gain={GAIN_DB}dB")
    print(f"[cry] Sliding window every {HOP_SAMPLES/OUT_RATE:.2f}s — Ctrl+C to stop\n")

    p_rec = subprocess.Popen(arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p_sox = subprocess.Popen(sox_cmd, stdin=p_rec.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p_rec.stdout.close()  # let p_rec receive SIGPIPE if p_sox dies

    bytes_per_sample = 2   # 16-bit output from sox
    hop_bytes        = HOP_SAMPLES * bytes_per_sample

    ring_buf  = np.zeros(MODEL_SAMPLES, dtype=np.float32)

    state       = "no_cry"
    cry_start   = None
    nocry_start = None
    elapsed     = 0.0

    try:
        while True:
            # Read one hop of audio from the continuous pipe
            raw = b""
            while len(raw) < hop_bytes:
                chunk = p_sox.stdout.read(hop_bytes - len(raw))
                if not chunk:
                    raise RuntimeError("Audio stream ended unexpectedly.")
                raw += chunk

            hop_pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # Slide ring buffer: drop oldest HOP_SAMPLES, append new
            ring_buf = np.roll(ring_buf, -HOP_SAMPLES)
            ring_buf[-HOP_SAMPLES:] = hop_pcm

            cry_prob = run_inference(interp, inp_idx, out_idx, ring_buf)
            now      = time.time()

            # State machine
            if cry_prob >= CRY_PROB_THRESHOLD:
                nocry_start = None
                if cry_start is None:
                    cry_start = now
                elapsed = now - cry_start
                if state == "no_cry" and elapsed >= CRY_CONFIRM_SECONDS:
                    state = "crying"
                    print(
                        f"\n{'='*52}\n"
                        f"  *** BABY CRYING DETECTED ***   prob={cry_prob:.3f}\n"
                        f"{'='*52}"
                    )
            else:
                cry_start = None
                if nocry_start is None:
                    nocry_start = now
                elapsed = now - nocry_start
                if state == "crying" and elapsed >= SILENCE_CONFIRM_SECS:
                    state = "no_cry"
                    print(
                        f"\n{'='*52}\n"
                        f"  Cry stopped / quiet            prob={cry_prob:.3f}\n"
                        f"{'='*52}"
                    )

            bar_len = 25
            filled  = int(min(cry_prob / max(CRY_PROB_THRESHOLD, 0.001), 1.0) * bar_len)
            bar     = "#" * filled + "." * (bar_len - filled)
            lbl     = "CRYING" if state == "crying" else "quiet "
            thr_bar = "#" * bar_len  # threshold marker position = full bar
            print(
                f"\r[{bar}] prob={cry_prob:.3f}  [{lbl}]  confirm={elapsed:.1f}s   ",
                end="",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\n[cry] Stopped.")
    except Exception as e:
        print(f"\n[cry] Error: {e}")
    finally:
        p_sox.terminate()
        p_rec.terminate()

# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 52)
    print("  Waladi Cry Detection — startup")
    print("=" * 52)

    ensure_sox()
    ensure_tflite()
    ensure_model()

    print("[cry] Loading model...")
    interp, inp_idx, out_idx = load_model()
    print("[cry] Model ready")

    try:
        run_loop(interp, inp_idx, out_idx)
    except KeyboardInterrupt:
        print("\n[cry] Stopped.")

if __name__ == "__main__":
    main()
