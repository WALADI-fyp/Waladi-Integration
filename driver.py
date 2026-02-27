import subprocess
import sys
import signal
import os

SERVICES = [
    "services.sht31_service.main",
    "services.fusion_service.main",
    "services.api_service.main",
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

    print("[driver] all services running — stream at http://<pi-ip>:8000/stream")
    print("[driver] press Ctrl+C to stop\n")

    # Wait — if any service crashes, shut everything else down too
    while True:
        for p in processes:
            if p.poll() is not None:
                print(f"[driver] a service exited unexpectedly (pid={p.pid}), shutting down")
                shutdown(None, None)
