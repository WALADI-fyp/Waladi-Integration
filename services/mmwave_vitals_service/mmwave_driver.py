import re
import serial
import time
from typing import Optional, Tuple


class MmwaveVitalsSensor:
    """
    Reads breathing rate and heart rate from a serial-connected mmWave sensor.

    Expected output examples can be adapted inside parse_line():
      - 'Breathing rate: 18'
      - 'Heart rate: 72'
      - 'breathing_rate=18,heart_rate=72'
    """

    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 115200,
        timeout: float = 1.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None

        self._latest_breathing: Optional[float] = None
        self._latest_heart: Optional[float] = None

    def connect(self) -> None:
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )
        time.sleep(0.2)

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def parse_line(self, line: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse one line and update breathing / heart when present.
        Adjust regexes to match your proven script output.
        """
        breathing = None
        heart = None

        # Example 1: "Breathing rate: 18"
        m = re.search(r"breathing(?:\s*rate)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", line, re.IGNORECASE)
        if m:
            breathing = float(m.group(1))

        # Example 2: "Heart rate: 72"
        m = re.search(r"heart(?:\s*rate)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", line, re.IGNORECASE)
        if m:
            heart = float(m.group(1))

        return breathing, heart

    def read(self, max_wait_s: float = 5.0) -> Tuple[Optional[float], Optional[float]]:
        """
        Read until we get fresh values or timeout.
        Returns the latest known breathing and heart rate.
        """
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Sensor serial port is not connected")

        deadline = time.time() + max_wait_s

        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue

            breathing, heart = self.parse_line(line)

            if breathing is not None:
                self._latest_breathing = breathing
            if heart is not None:
                self._latest_heart = heart

            # Return once at least one meaningful value has been updated
            if breathing is not None or heart is not None:
                return self._latest_breathing, self._latest_heart

        return self._latest_breathing, self._latest_heart