import re
import time
import serial
from typing import Optional, Tuple

breath_re  = re.compile(r"breath_rate:\s*([0-9.]+)")
heart_re   = re.compile(r"heart_rate:\s*([0-9.]+)")


class MmwaveVitalsSensor:
    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        timeout: float = 1.0,
    ):
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self.ser: Optional[serial.Serial] = None

        self._latest_breath: Optional[float] = None
        self._latest_heart:  Optional[float] = None

    def connect(self) -> None:
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )
        time.sleep(2)  # let serial settle
        print(f"[mmwave] opened {self.port} at {self.baudrate}")

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def read(self, max_wait_s: float = 5.0) -> Tuple[Optional[float], Optional[float]]:
        """
        Read lines from serial until we get both breath_rate and heart_rate,
        or until max_wait_s seconds have passed.
        Returns (breathing_rate_bpm, heart_rate_bpm).
        """
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial port not connected")

        deadline = time.time() + max_wait_s
        got_breath = False
        got_heart  = False

        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            print(f"[mmwave] RAW: {line}")

            m = breath_re.search(line)
            if m:
                self._latest_breath = float(m.group(1))
                got_breath = True

            m = heart_re.search(line)
            if m:
                self._latest_heart = float(m.group(1))
                got_heart = True

            if got_breath and got_heart:
                break

        return self._latest_breath, self._latest_heart
