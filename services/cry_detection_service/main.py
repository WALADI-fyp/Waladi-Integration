"""
cry_detection_service
─────────────────────
Continuously listens to the INMP441 mic, runs YAMNet TFLite
inference, and manages cry alert lifecycle:

  crying starts → INSERT cry_alerts row (ended_at = NULL)
                → publish  waladi/alerts/cry  {event: "cry_start", ...}

  crying stops  → UPDATE cry_alerts SET ended_at, duration_s
                → publish  waladi/alerts/cry  {event: "cry_end", ...}

Reads config from:
  config/audio.yaml   — mic, model, thresholds
  config/mqtt.yaml    — EMQX broker
  config/db.yaml      — TimescaleDB
  config/topics.yaml  — topic names

Run from project root (launched by driver.py):
  python3 -m services.cry_detection_service.main
"""

import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import yaml

from config.device import get_device_id
from shared.db_client import DbClient
from shared.message import make_message, now_ms
from shared.mqtt_client import MqttClient


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_interpreter_class():
    for mod in ["tflite_runtime.interpreter", "ai_edge_litert.interpreter"]:
        try:
            m = __import__(mod, fromlist=["Interpreter"])
            return m.Interpreter
        except ImportError:
            pass
    raise SystemExit(
        "[cry] Neither tflite-runtime nor ai-edge-litert is installed.\n"
        "  pip install tflite-runtime --break-system-packages"
    )


# ─────────────────────────────────────────────
#  Model loader
# ─────────────────────────────────────────────

def load_model(model_path: str):
    Interpreter = get_interpreter_class()
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"[cry] YAMNet model not found at {model_path}.\n"
            f"  Run cry_detection_standalone.py once to download it, or\n"
            f"  set audio.model_path in config/audio.yaml."
        )
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"[cry] Model loaded — input: {inp['shape']}  output: {out['shape']}")
    return interp, inp["index"], out["index"]


# ─────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────

def run_inference(interp, inp_idx: int, out_idx: int,
                  waveform, cry_class_ids: list, model_samples: int) -> float:
    import numpy as np

    n = len(waveform)
    if n >= model_samples:
        w = waveform[:model_samples]
    else:
        w = np.pad(waveform, (0, model_samples - n))

    interp.set_tensor(inp_idx, w)
    interp.invoke()

    scores = interp.get_tensor(out_idx)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    mean_scores = scores.mean(axis=0)
    return float(max(
        (mean_scores[i] for i in cry_class_ids if i < len(mean_scores)),
        default=0.0,
    ))


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────

def run_loop(*, interp, inp_idx: int, out_idx: int,
             audio_cfg: dict, mqtt: MqttClient, db: DbClient,
             cry_topic: str, device_id: str, user_id: str):

    import numpy as np

    card_device   = audio_cfg["card_device"]
    src_rate      = audio_cfg["src_rate"]
    src_channels  = audio_cfg["src_channels"]
    out_rate      = audio_cfg["out_rate"]
    gain_db       = audio_cfg["gain_db"]
    model_samples = audio_cfg["model_samples"]
    hop_samples   = audio_cfg["hop_samples"]
    cry_class_ids = audio_cfg["cry_class_ids"]
    threshold     = audio_cfg["cry_prob_threshold"]
    cry_confirm   = audio_cfg["cry_confirm_seconds"]
    sil_confirm   = audio_cfg["silence_confirm_secs"]

    # Continuous arecord → sox pipe
    # NOTE: gain -n (normalize) is intentionally omitted —
    # it buffers the entire stream before outputting anything.
    arecord_cmd = [
        "arecord",
        "-D", card_device,
        "-c", str(src_channels),
        "-r", str(src_rate),
        "-f", "S32_LE",
        "-t", "raw",
    ]
    sox_cmd = [
        "stdbuf", "-o0",
        "sox",
        "--buffer", "1024",
        "-t", "raw", "-b", "32", "-e", "signed-integer",
        "-r", str(src_rate), "-c", str(src_channels), "-",
        "-t", "raw", "-b", "16", "-e", "signed-integer",
        "-r", str(out_rate), "-c", "1", "-",
        "remix", "1",
        "gain", str(gain_db),
    ]

    p_rec = subprocess.Popen(arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p_sox = subprocess.Popen(sox_cmd, stdin=p_rec.stdout,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p_rec.stdout.close()

    hop_bytes = hop_samples * 2   # 16-bit = 2 bytes per sample
    ring_buf  = np.zeros(model_samples, dtype=np.float32)

    state        = "no_cry"
    cry_start    = None
    nocry_start  = None
    elapsed      = 0.0
    active_alert_id: int | None = None

    print(f"[cry] Listening on {card_device} — "
          f"threshold={threshold}  confirm={cry_confirm}s  gain={gain_db}dB")

    try:
        while True:
            # Read one hop from the continuous pipe
            raw = b""
            while len(raw) < hop_bytes:
                chunk = p_sox.stdout.read(hop_bytes - len(raw))
                if not chunk:
                    raise RuntimeError("Audio pipe closed unexpectedly.")
                raw += chunk

            hop_pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # Slide ring buffer
            ring_buf = np.roll(ring_buf, -hop_samples)
            ring_buf[-hop_samples:] = hop_pcm

            cry_prob = run_inference(interp, inp_idx, out_idx,
                                     ring_buf, cry_class_ids, model_samples)
            now = time.time()

            # ── State machine ──────────────────────────────────────────────
            if cry_prob >= threshold:
                nocry_start = None
                if cry_start is None:
                    cry_start = now
                elapsed = now - cry_start

                if state == "no_cry" and elapsed >= cry_confirm:
                    state = "crying"
                    ts = now_ms()
                    print(f"\n[cry] *** CRYING DETECTED *** prob={cry_prob:.3f}")

                    # DB — open alert
                    try:
                        active_alert_id = db.insert_cry_alert_start(
                            user_id=user_id,
                            device_id=device_id,
                            started_at_ms=ts,
                        )
                        print(f"[cry] alert #{active_alert_id} opened in DB")
                    except Exception as e:
                        print(f"[cry] DB insert failed: {e}")
                        active_alert_id = None

                    # MQTT — notify frontend live
                    mqtt.publish_json(cry_topic, make_message(
                        source="cry_detection_service",
                        data={
                            "event":     "cry_start",
                            "device_id": device_id,
                            "prob":      round(cry_prob, 3),
                            "alert_id":  active_alert_id,
                        },
                    ))

            else:
                cry_start = None
                if nocry_start is None:
                    nocry_start = now
                elapsed = now - nocry_start

                if state == "crying" and elapsed >= sil_confirm:
                    state = "no_cry"
                    ts = now_ms()
                    print(f"\n[cry] Crying stopped prob={cry_prob:.3f}")

                    # DB — close alert
                    if active_alert_id is not None:
                        try:
                            db.update_cry_alert_end(
                                alert_id=active_alert_id,
                                ended_at_ms=ts,
                            )
                            print(f"[cry] alert #{active_alert_id} closed in DB")
                        except Exception as e:
                            print(f"[cry] DB update failed: {e}")
                        active_alert_id = None

                    # MQTT
                    mqtt.publish_json(cry_topic, make_message(
                        source="cry_detection_service",
                        data={
                            "event":     "cry_end",
                            "device_id": device_id,
                            "prob":      round(cry_prob, 3),
                        },
                    ))



    except KeyboardInterrupt:
        pass
    finally:
        p_sox.terminate()
        p_rec.terminate()
        # If we crash mid-cry, close the open alert in DB
        if active_alert_id is not None:
            try:
                db.update_cry_alert_end(
                    alert_id=active_alert_id,
                    ended_at_ms=now_ms(),
                )
                print(f"\n[cry] alert #{active_alert_id} force-closed on shutdown")
            except Exception:
                pass


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main():
    audio_cfg = load_yaml("config/audio.yaml")["audio"]
    mqtt_cfg  = load_yaml("config/mqtt.yaml")
    db_cfg    = load_yaml("config/db.yaml")["timescale"]
    topics    = load_yaml("config/topics.yaml")["topics"]

    device_id  = get_device_id()
    cry_topic  = topics["cry_alert"]

    # ── DB ────────────────────────────────────────────────────────────
    db = DbClient(
        host=db_cfg["host"],
        port=db_cfg["port"],
        dbname=db_cfg["dbname"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        sslmode=db_cfg.get("sslmode", "require"),
    )
    db.connect()
    db.init_cry_alerts_table()

    # Resolve user_id (same pattern as db_writer_service)
    user_id = None
    for attempt in range(3):
        try:
            user_id = db.get_user_id(device_id)
        except Exception as e:
            print(f"[cry] DB error looking up user_id: {e}")
        if user_id:
            print(f"[cry] paired to user_id={user_id}")
            break
        print(f"[cry] device not yet paired (attempt {attempt+1}/3) — retrying in 5s")
        time.sleep(5)

    if not user_id:
        user_id = "unassigned"
        print("[cry] no pairing found — writing alerts with user_id='unassigned'")

    # ── MQTT ──────────────────────────────────────────────────────────
    mqtt = MqttClient(
        client_id=f"cry_detection_{device_id}",
        host=mqtt_cfg["broker"]["host"],
        port=mqtt_cfg["broker"]["port"],
        keepalive=mqtt_cfg["client"]["keepalive"],
        username=mqtt_cfg["broker"].get("username"),
        password=mqtt_cfg["broker"].get("password"),
        tls=mqtt_cfg["broker"].get("tls", False),
    )
    mqtt.connect()

    # ── Model ─────────────────────────────────────────────────────────
    interp, inp_idx, out_idx = load_model(audio_cfg["model_path"])

    print(f"[cry] service ready — publishing to '{cry_topic}'")

    try:
        run_loop(
            interp=interp,
            inp_idx=inp_idx,
            out_idx=out_idx,
            audio_cfg=audio_cfg,
            mqtt=mqtt,
            db=db,
            cry_topic=cry_topic,
            device_id=device_id,
            user_id=user_id,
        )
    finally:
        mqtt.close()
        db.close()


if __name__ == "__main__":
    main()
