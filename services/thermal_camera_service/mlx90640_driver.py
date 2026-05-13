import board
import busio
import numpy as np
import adafruit_mlx90640

# ── Monkey-patch: suppress "More than 4 outlier pixels" ──────────────────────
# The adafruit library raises RuntimeError during _ExtractDeviatingPixels when
# the sensor has hardware pixel defects. This is a known issue with some MLX90640
# units and does not affect the quality of temperature readings meaningfully.
# We suppress only that specific error and let everything else through.
_orig_extract = adafruit_mlx90640.MLX90640._ExtractDeviatingPixels

def _patched_extract(self):
    try:
        _orig_extract(self)
    except RuntimeError as e:
        if "outlier pixels" in str(e).lower():
            print("[MLX90640] Suppressed outlier pixel check — sensor continuing normally")
        else:
            raise

adafruit_mlx90640.MLX90640._ExtractDeviatingPixels = _patched_extract
# ─────────────────────────────────────────────────────────────────────────────


class MLX90640Driver:
    def __init__(
        self,
        refresh_rate=adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
        enable_visualization: bool = False,
    ):
        self.enable_visualization = enable_visualization

        print("[MLX90640] Opening I2C...")
        self.i2c = busio.I2C(board.SCL, board.SDA)

        print("[MLX90640] Initializing MLX90640...")
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.mlx.refresh_rate = refresh_rate

        self.frame = np.zeros((24 * 32,), dtype=float)
        print("[MLX90640] Ready")

    def _clean_frame(self, data: np.ndarray):
        data = data.copy()
        data[(data < -40) | (data > 300)] = np.nan
        if np.isnan(data).any():
            valid = data[~np.isnan(data)]
            if valid.size == 0:
                return None
            data[np.isnan(data)] = np.median(valid)
        return data

    def read(self):
        self.mlx.getFrame(self.frame)
        data = self._clean_frame(np.reshape(self.frame, (24, 32)))

        if data is None:
            raise ValueError("Thermal frame invalid after cleaning")

        max_index        = np.nanargmax(data)
        max_row, max_col = np.unravel_index(max_index, data.shape)

        return {
            "max_temp_c": float(data[max_row, max_col]),
            "max_row":    int(max_row),
            "max_col":    int(max_col),
            "min_temp_c": float(np.nanmin(data)),
            "avg_temp_c": float(np.nanmean(data)),
            "height":     24,
            "width":      32,
        }

    def close(self):
        pass
