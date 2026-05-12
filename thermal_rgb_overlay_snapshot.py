#!/usr/bin/env python3
"""
Standalone RGB + MLX90640 thermal snapshot overlay for Raspberry Pi.

Captures one normal camera snapshot and one thermal frame, creates a heatmap,
overlays the heatmap on the RGB image, saves the outputs, and optionally shows
an OpenCV preview window.

Examples:
  python3 thermal_rgb_overlay_snapshot.py
  python3 thermal_rgb_overlay_snapshot.py --alpha 0.45 --show
  python3 thermal_rgb_overlay_snapshot.py --camera-url http://localhost:8001/snapshot --show
  python3 thermal_rgb_overlay_snapshot.py --flip-thermal-x --rotate-thermal 180 --alpha 0.55

Notes:
  - If your existing camera_service is already running, the Pi camera may be busy.
    In that case use --camera-url http://localhost:8001/snapshot, or stop the service.
  - MLX90640 and RGB cameras do not have the same lens/FOV, so perfect alignment
    requires calibration. Use the crop/flip/rotate options below to tune alignment.
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
    p = argparse.ArgumentParser(description="Capture RGB + MLX90640 and create thermal overlay.")

    # Output / display
    p.add_argument("--output-dir", default="thermal_overlay_outputs", help="Directory for saved images")
    p.add_argument("--prefix", default=None, help="Filename prefix; default uses timestamp")
    p.add_argument("--show", action="store_true", help="Show output in an OpenCV window")
    p.add_argument("--wait-key-ms", type=int, default=0, help="cv2.waitKey duration when --show is used; 0 waits forever")

    # RGB source
    p.add_argument("--camera-url", default=None, help="Use existing HTTP JPEG snapshot endpoint instead of opening Picamera2")
    p.add_argument("--rgb-width", type=int, default=1280, help="Picamera2 capture width")
    p.add_argument("--rgb-height", type=int, default=720, help="Picamera2 capture height")
    p.add_argument("--settle-sec", type=float, default=0.7, help="Camera settle time before capture")
    p.add_argument("--jpeg-quality", type=int, default=90, help="Saved JPEG quality")

    # Thermal acquisition
    p.add_argument("--thermal-retries", type=int, default=8, help="MLX90640 read retry count")
    p.add_argument("--thermal-retry-delay", type=float, default=0.15, help="Delay between thermal retries")
    p.add_argument("--refresh-hz", type=int, default=4, choices=[1, 2, 4, 8, 16, 32, 64], help="MLX90640 refresh rate")

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
    p.add_argument("--colormap", default="inferno", choices=["inferno", "jet", "turbo", "hot", "magma", "plasma", "viridis"], help="Thermal colormap")
    p.add_argument("--min-temp", type=float, default=None, help="Fixed min temp C for normalization")
    p.add_argument("--max-temp", type=float, default=None, help="Fixed max temp C for normalization")
    p.add_argument("--percentile-low", type=float, default=2.0, help="Auto-normalization low percentile")
    p.add_argument("--percentile-high", type=float, default=98.0, help="Auto-normalization high percentile")
    p.add_argument("--draw-hotspot", action="store_true", default=True, help="Draw marker on hottest thermal pixel")
    p.add_argument("--no-draw-hotspot", dest="draw_hotspot", action="store_false")

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
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = resp.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode JPEG from {url}")
    return frame


def capture_rgb_picamera(width: int, height: int, settle_sec: float) -> np.ndarray:
    try:
        from picamera2 import Picamera2
        try:
            from libcamera import controls as libcontrols
        except Exception:
            libcontrols = None
    except ImportError as e:
        raise RuntimeError("picamera2/libcamera is not installed. Install it or use --camera-url.") from e

    picam2 = Picamera2()
    try:
        config = picam2.create_still_configuration(main={"size": (width, height), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        time.sleep(settle_sec)

        if libcontrols is not None:
            try:
                picam2.set_controls({"AfMode": libcontrols.AfModeEnum.Continuous})
                time.sleep(0.2)
            except Exception:
                pass

        rgb = picam2.capture_array("main")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass


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


def capture_thermal_frame(retries: int, retry_delay: float, refresh_hz: int) -> np.ndarray:
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
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = refresh_rate_constant(refresh_hz)
    frame = [0.0] * (THERMAL_H * THERMAL_W)

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            mlx.getFrame(frame)
            return clean_thermal_frame(np.array(frame, dtype=np.float32).reshape((THERMAL_H, THERMAL_W)))
        except Exception as e:
            last_error = e
            print(f"[thermal] read attempt {attempt}/{retries} failed: {e}")
            time.sleep(retry_delay)

    raise RuntimeError(f"Could not read MLX90640 after {retries} attempts: {last_error}")


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


def draw_annotations(img: np.ndarray, thermal: np.ndarray, t_min_norm: float, t_max_norm: float, args: argparse.Namespace) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    t_min = float(np.min(thermal))
    t_avg = float(np.mean(thermal))
    t_max = float(np.max(thermal))
    max_r, max_c = np.unravel_index(int(np.argmax(thermal)), thermal.shape)

    if args.draw_hotspot:
        x = int((max_c + 0.5) / thermal.shape[1] * w)
        y = int((max_r + 0.5) / thermal.shape[0] * h)
        cv2.drawMarker(out, (x, y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
        cv2.circle(out, (x, y), 10, (0, 0, 0), 2)
        cv2.putText(out, f"hot {t_max:.1f}C", (min(w - 160, x + 12), max(24, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(out, f"hot {t_max:.1f}C", (min(w - 160, x + 12), max(24, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.rectangle(out, (0, 0), (390, 92), (0, 0, 0), -1)
    lines = [
        f"MLX90640 min/avg/max: {t_min:.1f} / {t_avg:.1f} / {t_max:.1f} C",
        f"norm range: {t_min_norm:.1f}..{t_max_norm:.1f} C | alpha={args.alpha:.2f}",
        f"thermal grid after crop/orient: {thermal.shape[1]}x{thermal.shape[0]}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(out, text, (10, 24 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or datetime.now().strftime("%Y%m%d_%H%M%S")

    print("[rgb] capturing snapshot...")
    if args.camera_url:
        rgb = fetch_rgb_from_url(args.camera_url)
    else:
        rgb = capture_rgb_picamera(args.rgb_width, args.rgb_height, args.settle_sec)
    rgb = rotate_image(rgb, args.rotate_rgb)

    print("[thermal] capturing MLX90640 frame...")
    thermal = capture_thermal_frame(args.thermal_retries, args.thermal_retry_delay, args.refresh_hz)
    thermal = apply_thermal_orientation(thermal, args)
    thermal = crop_fraction(thermal, args.crop_left, args.crop_right, args.crop_top, args.crop_bottom)

    target_size = (rgb.shape[1], rgb.shape[0])
    heatmap, norm_min, norm_max = thermal_to_heatmap(thermal, args, target_size)
    alpha = max(0.0, min(1.0, args.alpha))
    overlay = cv2.addWeighted(rgb, 1.0 - alpha, heatmap, alpha, 0.0)
    overlay = draw_annotations(overlay, thermal, norm_min, norm_max, args)

    rgb_path = out_dir / f"{prefix}_rgb.jpg"
    heat_path = out_dir / f"{prefix}_thermal_heatmap.jpg"
    overlay_path = out_dir / f"{prefix}_overlay.jpg"
    npy_path = out_dir / f"{prefix}_thermal_celsius.npy"

    cv2.imwrite(str(rgb_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    cv2.imwrite(str(heat_path), heatmap, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    np.save(str(npy_path), thermal)

    print(f"[saved] RGB:     {rgb_path}")
    print(f"[saved] thermal: {heat_path}")
    print(f"[saved] overlay: {overlay_path}")
    print(f"[saved] raw C:   {npy_path}")

    if args.show:
        cv2.namedWindow("RGB + Thermal Overlay", cv2.WINDOW_NORMAL)
        cv2.imshow("RGB + Thermal Overlay", overlay)
        cv2.waitKey(args.wait_key_ms)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
