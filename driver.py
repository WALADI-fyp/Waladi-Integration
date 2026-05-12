import subprocess
import sys
import signal
import os
import time

SERVICES = [
    "services.sht31_service.main",
    "services.fusion_service.main",
    "services.camera_service.main",
    "services.db_writer_service.main",
    "services.mmwave_vitals_service.main",
    "services.thermal_camera_service.main",
    "services.ai_pose_service.main",
    "services.cry_detection_service.main",
    "services.sleep_detection_service.main",
]

processes = []


def shutdown(sig, frame):
    print("\n[driver] shutting down...")
    for p in processes:
        p.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


if __name__ == "__main__":
    print("[driver] starting Waladi backend...")

    for service in SERVICES:
        p = subprocess.Popen(
            [sys.executable, "-m", service],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        processes.append(p)
        print(f"[driver] started {service} (pid={p.pid})")

    print("\n[driver] all services running — only errors and key events will print\n")

    while True:
        for p in processes:
            if p.poll() is not None:
                # Find which service crashed
                idx = processes.index(p)
                name = SERVICES[idx] if idx < len(SERVICES) else "unknown"
                print(f"[driver] ERROR: {name} exited unexpectedly (pid={p.pid}), shutting down")
                shutdown(None, None)
        time.sleep(1)
