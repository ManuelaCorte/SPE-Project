# Dynamic time warping implementation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from typing import Literal
from src.utils import Matrix, Float, remove_nans
from src.structs import PlotOptions


class DynamicTimeWarping:
    def __init__(
        self, t: Matrix[Literal["N"], Float], s: Matrix[Literal["M"], Float]
    ) -> None:
        if t.ndim != 1 or s.ndim != 1:
            raise ValueError("Both time series must be 1D")

        self.t: Matrix[Literal["N"], Float] = remove_nans(t)
        self.s: Matrix[Literal["M"], Float] = remove_nans(s)
        self._dist_matrix: Matrix[Literal["N M"], Float] = self._compute_dist_matrix()

    @property
    def cost_matrix(self) -> Matrix[Literal["N M"], Float]:
        if not hasattr(self, "_cost_matrix"):
            raise AttributeError(
                "Cost matrix has not been computed, run compute_distance first."
            )
        return self._cost_matrix

    @property
    def warp_path(self) -> list[tuple[int, int]]:
        if not hasattr(self, "_warp_path"):
            raise AttributeError(
                "Warp path has not been computed, run compute_distance first."
            )
        return self._warp_path

    def _compute_dist_matrix(self) -> Matrix[Literal["N M"], Float]:
        n = len(self.t)
        m = len(self.s)
        dist_matrix: Matrix[Literal["N M"], Float] = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                dist_matrix[i, j] = (self.t[i] - self.s[j]) ** 2

        return dist_matrix

    def compute_distance(self) -> float:
        n, m = self._dist_matrix.shape

        dtw: Matrix[Literal["N M"], Float] = np.zeros((n, m))
        dtw[0, 0] = self._dist_matrix[0, 0]

        # Initialize the first row
        for i in range(1, n):
            dtw[i, 0] = self._dist_matrix[i, 0] + dtw[i - 1, 0]

        # Initialize the first column
        for j in range(1, m):
            dtw[0, j] = self._dist_matrix[0, j] + dtw[0, j - 1]

        prev_values: dict[tuple[int, int], tuple[int, int]] = {}
        for i in range(1, n):
            for j in range(1, m):
                cost = self._dist_matrix[i, j]
                next_values: list[tuple[float, tuple[int, int]]] = [
                    (dtw[i - 1, j], (i - 1, j)),
                    (dtw[i, j - 1], (i, j - 1)),
                    (dtw[i - 1, j - 1], (i - 1, j - 1)),
                ]
                min_cost, prev_idx = min(next_values, key=lambda x: x[0])
                dtw[i, j] = cost + min_cost
                prev_values[(i, j)] = prev_idx

        path: list[tuple[int, int]] = []
        i, j = n - 1, m - 1
        while i > 0 and j > 0:
            path.append((i, j))
            i, j = prev_values[(i, j)]
        path.append((0, 0))
        path.reverse()

        self._cost_matrix = dtw
        self._warp_path = path
        dist: float = dtw[-1, -1]
        return dist

    def plot(self, args: PlotOptions) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax = sbn.heatmap(
            self.cost_matrix,
            square=True,
            linewidths=0.1,
            cmap="YlGnBu",
            ax=ax,
        )
        ax.invert_yaxis()
        ax.set_title(args.title)
        ax.set_xlabel(args.x_axis)
        ax.set_ylabel(args.y_axis)

        # Get the warp path in x and y directions
        path_x = [p[0] for p in self.warp_path]
        path_y = [p[1] for p in self.warp_path]

        # Align the path from the center of each cell
        path_xx = [x + 0.5 for x in path_x]
        path_yy = [y + 0.5 for y in path_y]

        ax.plot(path_yy, path_xx, color="red", linewidth=3, alpha=0.2)

        if args.save:
            fig.savefig(f"data/results/plots/{args.filename}.png", dpi=300)
