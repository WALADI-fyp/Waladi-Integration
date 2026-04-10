import serial
import time
from typing import Optional


class MmwaveVitalsSensor:
    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 115200,
        timeout: float = 0.2,
        debug: bool = True,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.debug = debug
        self.ser: Optional[serial.Serial] = None
        self.buffer = bytearray()

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

    def read_raw_chunks(self):
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Sensor serial port is not connected")

        data = self.ser.read(128)
        if data:
            self.buffer.extend(data)
            if self.debug:
                print("[RAW HEX]", " ".join(f"{b:02X}" for b in data))

    def extract_frames(self):
        """
        Very simple frame finder for debugging.
        Looks for FF 01 and tries to split until next FF 01.
        """
        frames = []

        while True:
            start = self.buffer.find(b"\xFF\x01")
            if start == -1:
                if len(self.buffer) > 2048:
                    self.buffer = self.buffer[-256:]
                break

            next_start = self.buffer.find(b"\xFF\x01", start + 2)
            if next_start == -1:
                if start > 0:
                    del self.buffer[:start]
                break

            frame = bytes(self.buffer[start:next_start])
            frames.append(frame)
            del self.buffer[:next_start]

        return frames

    def debug_read_frames(self, duration_s: float = 5.0):
        end_time = time.time() + duration_s
        while time.time() < end_time:
            self.read_raw_chunks()
            frames = self.extract_frames()
            for frame in frames:
                print("[FRAME]", " ".join(f"{b:02X}" for b in frame))
