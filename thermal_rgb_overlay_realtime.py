#!/usr/bin/env python3
"""
Realtime RGB + MLX90640 thermal overlay feed for Raspberry Pi.

This is the realtime version of the snapshot overlay script:
- opens a live OpenCV window
- captures RGB frames from camera_service /snapshot or directly from Picamera2
- captures MLX90640 thermal frames
- overlays the thermal heatmap on the RGB feed
- displays the current max temperature at the top-right
- optionally saves a frame when you press "s"

Run:
  python3 thermal_rgb_overlay_realtime.py

Useful:
  python3 thermal_rgb_overlay_realtime.py --camera-url http://localhost:8001/snapshot
  python3 thermal_rgb_overlay_realtime.py --camera-url none
  python3 thermal_rgb_overlay_realtime.py --alpha 0.45 --rotate-thermal 180 --flip-thermal-x

Keys:
  q or ESC  quit
  s         save current overlay frame
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


THERMAL_H = 24
THERMAL_W = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime RGB + MLX90640 thermal overlay feed.")

    # Display / output
    p.add_argument("--window-name", default="Waladi RGB + Thermal Realtime", help="OpenCV window title")
    p.add_argument("--output-dir", default="thermal_realtime_captures", help="Directory used when pressing 's'")
    p.add_argument("--display-width", type=int, default=1280, help="Resize displayed feed to this width; 0 keeps source width")
    p.add_argument("--target-fps", type=float, default=4.0, help="Display loop target FPS")

    # RGB source
    p.add_argument(
        "--camera-url",
        default="auto",
        help="'auto' tries http://localhost:8001/snapshot first, 'none' opens Picamera2 directly, or pass a JPEG URL",
    )
    p.add_argument("--rgb-width", type=int, default=2304, help="Picamera2 capture width")
    p.add_argument("--rgb-height", type=int, default=1296, help="Picamera2 capture height")
    p.add_argument("--settle-sec", type=float, default=2.0, help="Picamera2 settle time")
    p.add_argument("--warmup-frames", type=int, default=8, help="Discard initial Picamera2 frames")
    p.add_argument("--debug-camera", action="store_true", help="Print Picamera2 metadata")

    # Thermal acquisition
    p.add_argument("--thermal-retries", type=int, default=8, help="MLX90640 read retry count")
    p.add_argument("--thermal-retry-delay", type=float, default=0.08, help="Delay between thermal read retries")
    p.add_argument(
        "--refresh-hz",
        type=int,
        default=4,
        choices=[1, 2, 4, 8, 16, 32, 64],
        help="MLX90640 refresh rate",
    )

    # Alignment controls
    p.add_argument("--rotate-rgb", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate RGB image")
    p.add_argument("--rotate-thermal", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate thermal matrix/heatmap")
    p.add_argument("--flip-thermal-x", action="store_true", help="Flip thermal heatmap left/right")
    p.add_argument("--flip-thermal-y", action="store_true", help="Flip thermal heatmap up/down")
    p.add_argument("--crop-left", type=float, default=0.0, help="Crop thermal overlay fraction from left, 0..0.45")
    p.add_argument("--crop-right", type=float, default=0.0, help="Crop thermal overlay fraction from right, 0..0.45")
    p.add_argument("--crop-top", type=float, default=0.0, help="Crop thermal overlay fraction from top, 0..0.45")
    p.add_argument("--crop-bottom", type=float, default=0.0, help="Crop thermal overlay fraction from bottom, 0..0.45")

    # Visualization
    p.add_argument("--alpha", type=float, default=0.50, help="Thermal overlay opacity, 0..1")
    p.add_argument(
        "--colormap",
        default="inferno",
        choices=["inferno", "jet", "turbo", "hot", "magma", "plasma", "viridis"],
        help="Thermal colormap",
    )
    p.add_argument("--min-temp", type=float, default=None, help="Fixed min temp C for normalization")
    p.add_argument("--max-temp", type=float, default=None, help="Fixed max temp C for normalization")
    p.add_argument("--percentile-low", type=float, default=2.0, help="Auto-normalization low percentile")
    p.add_argument("--percentile-high", type=float, default=98.0, help="Auto-normalization high percentile")
    p.add_argument("--draw-hotspot", action="store_true", default=True, help="Draw marker on hottest thermal pixel")
    p.add_argument("--no-draw-hotspot", dest="draw_hotspot", action="store_false")
    p.add_argument("--show-min-avg", action="store_true", help="Also show min/avg temperature in the top-left")

    return p.parse_args()


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def fetch_rgb_from_url(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=2) as resp:
        data = resp.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode JPEG from {url}")
    return frame


class RGBSource:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.url: Optional[str] = None
        self.picam2 = None
        self.libcontrols = None

    def open(self) -> None:
        camera_url = self.args.camera_url

        if camera_url == "auto":
            test_url = "http://localhost:8001/snapshot"
            try:
                _ = fetch_rgb_from_url(test_url)
                self.url = test_url
                print(f"[rgb] using existing camera_service: {test_url}")
                return
            except Exception as e:
                print(f"[rgb] camera_service not available ({e}); opening Picamera2 directly")
                camera_url = "none"

        if camera_url and camera_url != "none":
            self.url = camera_url
            print(f"[rgb] using URL: {self.url}")
            return

        try:
            from picamera2 import Picamera2
            try:
                from libcamera import controls as libcontrols
            except Exception:
                libcontrols = None
        except ImportError as e:
            raise RuntimeError("picamera2/libcamera is not installed. Use --camera-url http://localhost:8001/snapshot") from e

        self.libcontrols = libcontrols
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (self.args.rgb_width, self.args.rgb_height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(self.args.settle_sec)

        if self.libcontrols is not None:
            try:
                self.picam2.set_controls({"AfMode": self.libcontrols.AfModeEnum.Continuous})
            except Exception:
                pass

        try:
            props = self.picam2.camera_properties
            if "ScalerCropMaximum" in props:
                self.picam2.set_controls({"ScalerCrop": props["ScalerCropMaximum"]})
        except Exception:
            pass

        for _ in range(max(1, int(self.args.warmup_frames))):
            self.picam2.capture_array("main")
            time.sleep(0.04)

        if self.args.debug_camera:
            try:
                print(f"[rgb] metadata: {self.picam2.capture_metadata()}")
            except Exception:
                pass

        print("[rgb] Picamera2 opened directly")

    def read(self) -> np.ndarray:
        if self.url:
            return fetch_rgb_from_url(self.url)

        if self.picam2 is None:
            raise RuntimeError("RGB source is not open")

        frame = self.picam2.capture_array("main")
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None


def patch_mlx90640_outlier_error() -> None:
    import adafruit_mlx90640

    if getattr(adafruit_mlx90640.MLX90640, "_waladi_outlier_patch", False):
        return

    original = adafruit_mlx90640.MLX90640._ExtractDeviatingPixels

    def patched(self):
        try:
            original(self)
        except RuntimeError as e:
            if "outlier pixels" in str(e).lower():
                print("[MLX90640] Suppressed outlier pixel check; continuing")
            else:
                raise

    adafruit_mlx90640.MLX90640._ExtractDeviatingPixels = patched
    adafruit_mlx90640.MLX90640._waladi_outlier_patch = True


def refresh_rate_constant(refresh_hz: int):
    import adafruit_mlx90640

    mapping = {
        1: adafruit_mlx90640.RefreshRate.REFRESH_1_HZ,
        2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
        4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
        8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
        16: adafruit_mlx90640.RefreshRate.REFRESH_16_HZ,
        32: adafruit_mlx90640.RefreshRate.REFRESH_32_HZ,
        64: adafruit_mlx90640.RefreshRate.REFRESH_64_HZ,
    }
    return mapping[refresh_hz]


def clean_thermal_frame(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32).copy()
    data[(data < -40.0) | (data > 300.0)] = np.nan
    if np.isnan(data).any():
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            raise RuntimeError("Thermal frame is fully invalid")
        data[np.isnan(data)] = np.median(valid)
    return data


class ThermalSource:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mlx = None
        self.frame = [0.0] * (THERMAL_H * THERMAL_W)

    def open(self) -> None:
        try:
            import board
            import busio
            import adafruit_mlx90640
        except ImportError as e:
            raise RuntimeError(
                "Thermal dependencies missing. Install with: "
                "pip3 install adafruit-circuitpython-mlx90640 numpy opencv-python"
            ) from e

        patch_mlx90640_outlier_error()

        i2c = busio.I2C(board.SCL, board.SDA)
        self.mlx = adafruit_mlx90640.MLX90640(i2c)
        self.mlx.refresh_rate = refresh_rate_constant(self.args.refresh_hz)
        print(f"[thermal] MLX90640 opened at {self.args.refresh_hz} Hz")

    def read(self) -> np.ndarray:
        if self.mlx is None:
            raise RuntimeError("Thermal source is not open")

        last_error: Optional[Exception] = None
        for attempt in range(1, self.args.thermal_retries + 1):
            try:
                self.mlx.getFrame(self.frame)
                data = np.array(self.frame, dtype=np.float32).reshape((THERMAL_H, THERMAL_W))
                return clean_thermal_frame(data)
            except Exception as e:
                last_error = e
                if attempt == self.args.thermal_retries:
                    raise RuntimeError(f"Could not read MLX90640: {last_error}") from e
                time.sleep(self.args.thermal_retry_delay)

        raise RuntimeError(f"Could not read MLX90640: {last_error}")


def apply_thermal_orientation(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    out = data.copy()
    if args.rotate_thermal == 90:
        out = np.rot90(out, k=3)
    elif args.rotate_thermal == 180:
        out = np.rot90(out, k=2)
    elif args.rotate_thermal == 270:
        out = np.rot90(out, k=1)
    if args.flip_thermal_x:
        out = np.fliplr(out)
    if args.flip_thermal_y:
        out = np.flipud(out)
    return out


def crop_fraction(data: np.ndarray, left: float, right: float, top: float, bottom: float) -> np.ndarray:
    h, w = data.shape[:2]
    left = max(0.0, min(0.45, left))
    right = max(0.0, min(0.45, right))
    top = max(0.0, min(0.45, top))
    bottom = max(0.0, min(0.45, bottom))
    x1 = int(round(w * left))
    x2 = int(round(w * (1.0 - right)))
    y1 = int(round(h * top))
    y2 = int(round(h * (1.0 - bottom)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop settings removed the whole thermal image")
    return data[y1:y2, x1:x2]


def colormap_id(name: str) -> int:
    return {
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "hot": cv2.COLORMAP_HOT,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }[name]


def thermal_to_heatmap(data: np.ndarray, args: argparse.Namespace, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, float]:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise RuntimeError("No valid thermal values")

    t_min = args.min_temp if args.min_temp is not None else float(np.percentile(valid, args.percentile_low))
    t_max = args.max_temp if args.max_temp is not None else float(np.percentile(valid, args.percentile_high))
    if t_max <= t_min:
        t_max = t_min + 1.0

    norm = np.clip((data - t_min) / (t_max - t_min), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    heat_small = cv2.applyColorMap(gray, colormap_id(args.colormap))
    heat_big = cv2.resize(heat_small, target_size, interpolation=cv2.INTER_CUBIC)
    return heat_big, t_min, t_max


def classify_temperature(temp_c: float) -> Tuple[str, Tuple[int, int, int]]:
    """
    Returns label and BGR text color.
    """
    if temp_c > 38.0:
        return "SEVERE", (255, 255, 255)
    if temp_c > 37.5:
        return "MODERATELY HIGH", (255, 255, 255)
    if temp_c > 37.0:
        return "NORMAL HIGH", (255, 255, 255)
    return "NORMAL", (255, 255, 255)


def draw_temperature_badge(img: np.ndarray, max_temp: float) -> None:
    h, w = img.shape[:2]
    label, text_color = classify_temperature(max_temp)
    line1 = f"{max_temp:.1f} C"
    line2 = label

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale1 = max(0.75, min(1.25, w / 1100.0))
    scale2 = max(0.45, min(0.75, w / 1700.0))
    thickness1 = 2
    thickness2 = 1

    (tw1, th1), _ = cv2.getTextSize(line1, font, scale1, thickness1)
    (tw2, th2), _ = cv2.getTextSize(line2, font, scale2, thickness2)
    box_w = max(tw1, tw2) + 36
    box_h = th1 + th2 + 42
    x1 = max(10, w - box_w - 16)
    y1 = 16
    x2 = w - 16
    y2 = y1 + box_h

    # Dark top-right badge.
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)

    cv2.putText(img, line1, (x1 + 18, y1 + th1 + 16), font, scale1, text_color, thickness1)
    cv2.putText(img, line2, (x1 + 18, y1 + th1 + th2 + 30), font, scale2, text_color, thickness2)


def draw_hotspot_marker(img: np.ndarray, thermal: np.ndarray) -> None:
    h, w = img.shape[:2]
    max_r, max_c = np.unravel_index(int(np.argmax(thermal)), thermal.shape)
    x = int((max_c + 0.5) / thermal.shape[1] * w)
    y = int((max_r + 0.5) / thermal.shape[0] * h)
    cv2.drawMarker(img, (x, y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
    cv2.circle(img, (x, y), 10, (0, 0, 0), 2)


def draw_extra_stats(img: np.ndarray, thermal: np.ndarray, fps: float) -> None:
    t_min = float(np.min(thermal))
    t_avg = float(np.mean(thermal))
    t_max = float(np.max(thermal))
    lines = [
        f"min/avg/max: {t_min:.1f} / {t_avg:.1f} / {t_max:.1f} C",
        f"fps: {fps:.1f}",
        "q: quit | s: save",
    ]

    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (390, 92), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, text in enumerate(lines):
        cv2.putText(img, text, (18, 32 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)


def make_overlay(rgb: np.ndarray, thermal_raw: np.ndarray, args: argparse.Namespace, fps: float) -> np.ndarray:
    rgb = rotate_image(rgb, args.rotate_rgb)

    if args.display_width and rgb.shape[1] != args.display_width:
        scale = args.display_width / float(rgb.shape[1])
        rgb = cv2.resize(rgb, (args.display_width, max(1, int(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA)

    thermal = apply_thermal_orientation(thermal_raw, args)
    thermal = crop_fraction(thermal, args.crop_left, args.crop_right, args.crop_top, args.crop_bottom)

    heatmap, _, _ = thermal_to_heatmap(thermal, args, (rgb.shape[1], rgb.shape[0]))
    alpha = max(0.0, min(1.0, args.alpha))
    overlay = cv2.addWeighted(rgb, 1.0 - alpha, heatmap, alpha, 0.0)

    if args.draw_hotspot:
        draw_hotspot_marker(overlay, thermal)

    draw_temperature_badge(overlay, float(np.max(thermal)))

    if args.show_min_avg:
        draw_extra_stats(overlay, thermal, fps)
    else:
        cv2.putText(
            overlay,
            "q: quit | s: save",
            (14, overlay.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
        )

    return overlay


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_source = RGBSource(args)
    thermal_source = ThermalSource(args)

    print("[startup] opening RGB source...")
    rgb_source.open()
    print("[startup] opening thermal source...")
    thermal_source.open()

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    last_frame_time = time.time()
    fps = 0.0
    frame_interval = 1.0 / max(0.1, args.target_fps)

    try:
        while True:
            loop_start = time.time()

            try:
                rgb = rgb_source.read()
                thermal = thermal_source.read()
                now = time.time()
                dt = max(1e-6, now - last_frame_time)
                last_frame_time = now
                fps = (0.85 * fps) + (0.15 * (1.0 / dt)) if fps > 0 else 1.0 / dt

                overlay = make_overlay(rgb, thermal, args, fps)
            except Exception as e:
                print(f"[loop] frame failed: {e}")
                time.sleep(0.2)
                continue

            cv2.imshow(args.window_name, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = out_dir / f"{ts}_realtime_overlay.jpg"
                cv2.imwrite(str(path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f"[saved] {path}")

            elapsed = time.time() - loop_start
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    finally:
        rgb_source.close()
        cv2.destroyAllWindows()
        print("[shutdown] closed realtime feed")


if __name__ == "__main__":
    main()
