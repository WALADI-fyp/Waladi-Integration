import board
import busio
import numpy as np
import matplotlib.pyplot as plt
import adafruit_mlx90640


class MLX90640Driver:
    def __init__(
        self,
        refresh_rate=adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
        enable_visualization: bool = True,
    ):
        self.enable_visualization = enable_visualization

        print("[MLX90640] Opening I2C...")
        self.i2c = busio.I2C(board.SCL, board.SDA)

        print("[MLX90640] Initializing MLX90640...")
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.mlx.refresh_rate = refresh_rate

        self.frame = np.zeros((24 * 32,), dtype=float)

        self.fig = None
        self.ax = None
        self.img = None

        if self.enable_visualization:
            self._init_plot()

    def _init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(
            np.zeros((24, 32)),
            cmap="inferno",
            interpolation="bilinear"
        )
        plt.colorbar(self.img, ax=self.ax)
        self.ax.set_title("MLX90640 Thermal View")

    def _clean_frame(self, data: np.ndarray):
        data = data.copy()

        # Remove clearly invalid values
        data[(data < -40) | (data > 300)] = np.nan

        if np.isnan(data).any():
            valid = data[~np.isnan(data)]
            if valid.size == 0:
                return None
            data[np.isnan(data)] = np.median(valid)

        return data

    def _update_plot(self, data: np.ndarray, max_row: int, max_col: int, max_temp: float):
        if not self.enable_visualization:
            return

        vmin = np.min(data)
        vmax = np.max(data)

        self.ax.clear()
        self.img = self.ax.imshow(
            data,
            cmap="inferno",
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax
        )
        self.ax.scatter(max_col, max_row, marker="x", s=100)
        self.ax.set_title(
            f"MLX90640 | Max: {max_temp:.1f} C at ({max_row}, {max_col})"
        )
        plt.pause(0.001)

    def read(self):
        self.mlx.getFrame(self.frame)

        data = np.reshape(self.frame, (24, 32)).copy()
        data = self._clean_frame(data)

        if data is None:
            raise ValueError("Thermal frame invalid after cleaning")

        max_index = np.nanargmax(data)
        max_row, max_col = np.unravel_index(max_index, data.shape)
        max_temp = float(data[max_row, max_col])

        min_temp = float(np.nanmin(data))
        avg_temp = float(np.nanmean(data))

        self._update_plot(data, max_row, max_col, max_temp)

        return {
            "max_temp_c": max_temp,
            "max_row": int(max_row),
            "max_col": int(max_col),
            "min_temp_c": min_temp,
            "avg_temp_c": avg_temp,
            "height": 24,
            "width": 32,
        }

    def close(self):
        if self.enable_visualization and self.fig is not None:
            plt.ioff()
            plt.close(self.fig)