import subprocess
import wave
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# -----------------------
# Settings
# -----------------------
CARD_DEVICE = "hw:2,0"     # INMP441 capture card
DURATION_SEC = 5
SRC_RATE = 48000
SRC_FORMAT = "S32_LE"
SRC_CHANNELS = 2

# Output audio settings
OUT_RATE = 16000

# IMPORTANT:
# - If USE_NORMALIZE is True, sox will normalize close to max already.
#   Then adding a huge GAIN_DB will clip.
# - Keep GAIN_DB small (0..6) when USE_NORMALIZE=True.
USE_NORMALIZE = True
GAIN_DB = 1  # recommended 0..6 with normalize; set 0 if you want only normalize

# Files
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
raw_wav = Path(f"raw_{ts}.wav")
final_wav = Path(f"final_{ts}.wav")
plot_png = Path(f"wave_{ts}.png")

# Set the correct HDMI for your monitor
PLAY_DEVICE = "plughw:0,0"   # or "plughw:1,0"


def run(cmd):
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def record_raw():
    run([
        "arecord",
        "-D", CARD_DEVICE,
        "-c", str(SRC_CHANNELS),
        "-r", str(SRC_RATE),
        "-f", SRC_FORMAT,
        "-t", "wav",
        str(raw_wav),
        "-d", str(DURATION_SEC)
    ])
    print("Saved:", raw_wav)


def fix_and_process():
    # Force standard PCM WAV (NOT WAV_EXTENSIBLE) using -t wavpcm
    # Also force 16-bit output with -b 16 so Python wave can read it.
    sox_cmd = [
        "sox",
        str(raw_wav),
        "-t", "wavpcm",
        "-b", "16",
        str(final_wav),
        "remix", "1",                 # take LEFT channel
    ]

    if USE_NORMALIZE:
        sox_cmd += ["gain", "-n"]     # normalize safely

    if GAIN_DB != 0:
        sox_cmd += ["gain", str(GAIN_DB)]

    sox_cmd += [
        "rate", str(OUT_RATE),
        "channels", "1"               # mono
    ]

    run(sox_cmd)
    print("Saved:", final_wav)


def load_wav_16bit_mono(path: Path):
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if ch != 1 or sw != 2:
        raise RuntimeError(
            f"Expected 16-bit mono WAV. Got channels={ch}, sampwidth={sw} bytes."
        )

    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def plot_wave(x, sr):
    t = np.arange(len(x)) / sr

    plt.figure(figsize=(10, 4))
    plt.plot(t, x)
    plt.title("INMP441 waveform (fixed + amplified)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_png, dpi=200)
    print("Saved plot:", plot_png)




if __name__ == "__main__":
    record_raw()
    fix_and_process()

    x, sr = load_wav_16bit_mono(final_wav)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2)))
    print(f"Final WAV stats: sr={sr} Hz, peak={peak:.3f}, rms={rms:.3f}")

    plot_wave(x, sr)

 
