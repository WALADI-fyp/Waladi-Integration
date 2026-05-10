#!/usr/bin/env python3
"""
Waladi Baby Cry Detection — standalone single-file script
==========================================================
Handles everything: dependency checks, model download,
conversion (TF Hub → TFLite → ONNX → NCNN), and real-time detection.

Usage:
    python3 cry_detection_standalone.py

Edit the CONFIG block below to match your setup.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  CONFIG — edit these to match your Pi setup
# ══════════════════════════════════════════════════════════════════════

# Where model files live (or will be created if missing)
MODEL_DIR = Path.home() / "waladi_models" / "yamnet"

# INMP441 microphone ALSA device  (run `arecord -l` to confirm)
CARD_DEVICE   = "hw:2,0"
SRC_RATE      = 48000
SRC_FORMAT    = "S32_LE"
SRC_CHANNELS  = 2
OUT_RATE      = 16000
CHUNK_SECONDS = 1          # seconds of audio per inference call
GAIN_DB       = 7          # applied after normalization (keep ≤ 10)

# NCNN layer names — verify against your yamnet_opt.param first few lines
# Open it with: head -5 ~/waladi_models/yamnet/yamnet_opt.param
INPUT_NAME  = "serving_default_waveform:0"
OUTPUT_NAME = "PartitionedCall:0"

# YAMNet class IDs for crying (verify against yamnet_class_map.csv)
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
    """Install packages silently, skip if already present."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--break-system-packages", *packages],
        check=True,
    )


def ensure_sox():
    if shutil.which("sox") is None:
        print("[setup] sox not found — installing via apt...")
        subprocess.run(["sudo", "apt-get", "install", "-y", "-q", "sox"], check=True)


def ensure_ncnn_python():
    try:
        import ncnn  # noqa
    except ImportError:
        print("[setup] Installing ncnn Python package...")
        _pip("ncnn")


# ══════════════════════════════════════════════════════════════════════
#  MODEL SETUP  (TF Hub → TFLite → ONNX → NCNN)
# ══════════════════════════════════════════════════════════════════════

def _param_path() -> Path:
    return MODEL_DIR / "yamnet_opt.param"

def _bin_path() -> Path:
    return MODEL_DIR / "yamnet_opt.bin"


def model_exists() -> bool:
    return _param_path().exists() and _bin_path().exists()


def find_onnx2ncnn() -> str | None:
    """Search common locations for the onnx2ncnn binary."""
    candidates = [
        shutil.which("onnx2ncnn"),
        str(Path.home() / "ncnn/build/tools/onnx/onnx2ncnn"),
        "/usr/local/bin/onnx2ncnn",
        "/opt/ncnn/bin/onnx2ncnn",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


def find_ncnnoptimize() -> str | None:
    candidates = [
        shutil.which("ncnnoptimize"),
        str(Path.home() / "ncnn/build/tools/ncnnoptimize"),
        "/usr/local/bin/ncnnoptimize",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


def step1_export_tflite(out_dir: Path) -> Path:
    """Download YAMNet from TF Hub and export to float32 TFLite."""
    tflite_path = out_dir / "yamnet_float32.tflite"
    if tflite_path.exists():
        print("[model] TFLite already exists, skipping step 1.")
        return tflite_path

    print("[model] Step 1/3 — Downloading YAMNet from TF Hub and exporting TFLite...")
    print("[model]   This may take a few minutes on first run (downloads ~30 MB).")

    _pip("tensorflow", "tensorflow-hub")
    import tensorflow as tf
    import tensorflow_hub as hub

    INPUT_LENGTH = 16000  # 1 second at 16 kHz

    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    @tf.function(input_signature=[tf.TensorSpec(shape=[INPUT_LENGTH], dtype=tf.float32)])
    def yamnet_fixed(waveform):
        scores, _, _ = yamnet(waveform)
        return scores

    concrete_func = yamnet_fixed.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    print(f"[model]   Saved: {tflite_path} ({len(tflite_model)/1e6:.1f} MB)")
    return tflite_path


def step2_tflite_to_onnx(tflite_path: Path, out_dir: Path) -> Path:
    """Convert TFLite → ONNX."""
    onnx_path = out_dir / "yamnet.onnx"
    if onnx_path.exists():
        print("[model] ONNX already exists, skipping step 2.")
        return onnx_path

    print("[model] Step 2/3 — Converting TFLite → ONNX...")
    _pip("tf2onnx", "onnx", "onnxruntime")

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", str(tflite_path),
        "--output",  str(onnx_path),
        "--opset",   "13",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[model] tf2onnx stderr:", result.stderr[-2000:])
        raise RuntimeError("TFLite → ONNX conversion failed.")

    print(f"[model]   Saved: {onnx_path}")
    return onnx_path


def step3_onnx_to_ncnn(onnx_path: Path, out_dir: Path):
    """Simplify ONNX, then convert to NCNN .param/.bin and optimize."""
    onnx2ncnn = find_onnx2ncnn()
    if onnx2ncnn is None:
        print(
            "\n[model] ERROR: onnx2ncnn binary not found.\n"
            "  Build it once on your Pi (or dev machine), then copy the binary:\n\n"
            "    git clone https://github.com/Tencent/ncnn.git\n"
            "    cd ncnn && mkdir build && cd build\n"
            "    cmake -DNCNN_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release ..\n"
            "    make -j4\n"
            "    # binary is at:  ncnn/build/tools/onnx/onnx2ncnn\n"
            "    sudo cp ncnn/build/tools/onnx/onnx2ncnn /usr/local/bin/\n"
            "    sudo cp ncnn/build/tools/ncnnoptimize  /usr/local/bin/\n\n"
            "  Then re-run this script.\n"
        )
        sys.exit(1)

    # Simplify ONNX
    print("[model] Step 3/3 — Simplifying ONNX and converting to NCNN...")
    sim_path = out_dir / "yamnet_sim.onnx"
    try:
        _pip("onnx-simplifier")
        import onnx
        from onnxsim import simplify
        model = onnx.load(str(onnx_path))
        model_sim, ok = simplify(model)
        assert ok
        onnx.save(model_sim, str(sim_path))
        print("[model]   ONNX simplified OK")
    except Exception as e:
        print(f"[model]   onnxsim failed ({e}), using original ONNX")
        shutil.copy(onnx_path, sim_path)

    # onnx2ncnn
    raw_param = out_dir / "yamnet_raw.param"
    raw_bin   = out_dir / "yamnet_raw.bin"
    result = subprocess.run(
        [onnx2ncnn, str(sim_path), str(raw_param), str(raw_bin)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("[model] onnx2ncnn stderr:", result.stderr[-2000:])
        raise RuntimeError("ONNX → NCNN conversion failed.")
    print("[model]   Raw NCNN .param/.bin generated")

    # ncnnoptimize (fp16 weights — halves .bin size, still runs fp32 at inference)
    ncnnopt = find_ncnnoptimize()
    param_out = _param_path()
    bin_out   = _bin_path()

    if ncnnopt:
        result = subprocess.run(
            [ncnnopt, str(raw_param), str(raw_bin),
             str(param_out), str(bin_out), "65536"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"[model]   Optimised → {param_out.name}, {bin_out.name}")
            raw_param.unlink(missing_ok=True)
            raw_bin.unlink(missing_ok=True)
        else:
            print("[model]   ncnnoptimize failed, using raw files")
            shutil.move(str(raw_param), str(param_out))
            shutil.move(str(raw_bin),   str(bin_out))
    else:
        print("[model]   ncnnoptimize not found, using raw (slightly larger) files")
        shutil.move(str(raw_param), str(param_out))
        shutil.move(str(raw_bin),   str(bin_out))

    print(f"[model]   .param: {param_out} ({param_out.stat().st_size/1024:.0f} KB)")
    print(f"[model]   .bin:   {bin_out} ({bin_out.stat().st_size/1e6:.1f} MB)")


def ensure_model():
    """Full model pipeline: check → download → convert."""
    if model_exists():
        print(f"[model] Found existing model at {MODEL_DIR}")
        return

    print(f"[model] Model not found at {MODEL_DIR} — starting conversion pipeline...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tflite = step1_export_tflite(MODEL_DIR)
    onnx   = step2_tflite_to_onnx(tflite, MODEL_DIR)
    step3_onnx_to_ncnn(onnx, MODEL_DIR)

    if not model_exists():
        raise RuntimeError("Model conversion finished but .param/.bin not found.")

    print("[model] ✅ Model ready.")


# ══════════════════════════════════════════════════════════════════════
#  AUDIO CAPTURE  (arecord → sox → 16-bit mono float32)
# ══════════════════════════════════════════════════════════════════════

def capture_chunk(tmp_dir: Path) -> "np.ndarray":
    import numpy as np

    raw_wav   = tmp_dir / "raw.wav"
    final_wav = tmp_dir / "final.wav"

    # Record via arecord
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

    # sox: left channel only → normalize → gain → 16kHz mono 16-bit
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

    return (
        __import__("numpy")
        .frombuffer(raw, dtype="int16")
        .astype("float32") / 32768.0
    )


# ══════════════════════════════════════════════════════════════════════
#  NCNN INFERENCE
# ══════════════════════════════════════════════════════════════════════

def load_model():
    import ncnn
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads        = 4
    net.load_param(str(_param_path()))
    net.load_model(str(_bin_path()))
    return net


def run_inference(net, waveform) -> float:
    import ncnn
    import numpy as np

    mat_in = ncnn.Mat(waveform)
    ex     = net.create_extractor()
    ex.input(INPUT_NAME, mat_in)

    ret, mat_out = ex.extract(OUTPUT_NAME)
    if ret != 0:
        print(
            f"\n[cry] Inference error (ret={ret}). "
            f"Check INPUT_NAME / OUTPUT_NAME at the top of this script.\n"
            f"  Hint: run:  head -5 {_param_path()}"
        )
        return 0.0

    scores = np.array(mat_out)
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
            f"  🚨  BABY CRYING DETECTED   (prob={cry_prob:.3f})\n"
            f"{'='*52}"
        )
    else:
        print(
            f"\n{'='*52}\n"
            f"  ✅  Cry stopped / quiet    (prob={cry_prob:.3f})\n"
            f"{'='*52}"
        )


def run_loop(net):
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
            # ── capture ──────────────────────────────────────
            try:
                waveform = capture_chunk(tmp_dir)
            except subprocess.CalledProcessError as e:
                print(f"\n[cry] Audio capture failed: {e} — retrying in 2s")
                time.sleep(2)
                continue

            # ── infer ─────────────────────────────────────────
            cry_prob = run_inference(net, waveform)
            now      = time.time()

            # ── state machine ─────────────────────────────────
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

            # ── status bar ────────────────────────────────────
            bar_len = 20
            filled  = int(min(cry_prob, 1.0) * bar_len)
            bar     = "█" * filled + "░" * (bar_len - filled)
            lbl     = "🚨 CRYING" if state == "crying" else "🔇 quiet "
            print(
                f"\r[{bar}] prob={cry_prob:.3f}  {lbl}  "
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

    # 1. System tools
    ensure_sox()
    ensure_ncnn_python()

    # 2. Model (download + convert if needed)
    ensure_model()

    # 3. Load NCNN model
    print("[cry] Loading NCNN model into memory...")
    import ncnn  # noqa — already ensured above
    net = load_model()
    print("[cry] Model ready.")

    # 4. Run
    try:
        run_loop(net)
    except KeyboardInterrupt:
        print("\n[cry] Stopped.")


if __name__ == "__main__":
    main()
