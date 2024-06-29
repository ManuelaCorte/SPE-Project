import os
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import statsmodels.graphics.tsaplots as tsa

from ._misc import remove_nans
from ._types import Float, Matrix


@dataclass(frozen=True)
class PlotOptions:
    """Utility class to gather together the options for a plot."""

    filename: str
    title: str
    x_axis: str
    y_axis: str
    labels: list[str]
    save: bool


def plot_time_series(
    x: Matrix[Literal["N"], np.str_],
    y: Matrix[Literal["N"], Float],
    window: int,
    args: PlotOptions,
) -> None:
    if window % 2 == 0:
        raise ValueError("Window must be odd")

    y = remove_nans(y)

    weights = np.repeat(1.0 / window, window)
    moving_average: Matrix[Literal["N"], np.float32] = np.convolve(y, weights, "valid")

    # pad the moving average with NaNs
    pad = window // 2
    moving_average: Matrix[Literal["N"], np.float32] = np.pad(
        moving_average,
        (pad,),
        mode="constant",
        constant_values=np.NaN,
    )
    moving_sd = np.sqrt(np.convolve(np.square(y - moving_average), weights, "valid"))
    moving_sd = np.pad(
        moving_sd,
        (pad,),
        mode="constant",
        constant_values=np.NaN,
    )

    _, ax = plt.subplots(1, 1, figsize=(20, 10))
    sbn.set_theme(style="darkgrid")
    sbn.lineplot(x=x, y=y, label="Original", ax=ax)
    sbn.lineplot(
        x=x, y=moving_average, label=f"Rolling average ({window})", linewidth=2, ax=ax
    )
    sbn.lineplot(x=x, y=moving_sd, label=f"Rolling SD ({window})", linewidth=2, ax=ax)
    ax.set_title(args.title)
    ax.set_xlabel(args.x_axis)
    ax.set_ylabel(args.y_axis)
    # insert 1 tick per year
    ax.set_xticks(range(0, len(x), 12))
    # show only the year
    ax.set_xticklabels([x[i][:4] for i in range(0, len(x), 12)], rotation=45)
    ax.legend()

    if args.save:
        if not os.path.exists("data/results/plots"):
            os.makedirs("data/results/plots")
        plt.savefig(f"plots/{args.filename}.png")


def acf_plot(
    x: Matrix[Literal["N"], np.str_],
    y: Matrix[Literal["N"], Float],
    lag: int,
    args: PlotOptions,
) -> None:
    y = remove_nans(y)

    acf_figure, ax = plt.subplots(1, 1, figsize=(20, 10))
    tsa.plot_acf(y, lags=lag, title=args.title, ax=ax)
    ax.set_xlabel(args.x_axis)
    ax.set_ylabel(args.y_axis)

    if args.save:
        if not os.path.exists("data/results/plots"):
            os.makedirs("data/results/plots")
        acf_figure.savefig(f"data/results/plots/{args.filename}.png")


def pacf_plot(
    x: Matrix[Literal["N"], np.str_],
    y: Matrix[Literal["N"], np.float32],
    lag: int,
    args: PlotOptions,
) -> None:
    y = remove_nans(y)

    pacf_figure, ax = plt.subplots(1, 1, figsize=(20, 10))
    tsa.plot_pacf(y, lags=lag, title=args.title, ax=ax)
    ax.set_xlabel(args.x_axis)
    ax.set_ylabel(args.y_axis)

    if args.save:
        if not os.path.exists("data/results/plots"):
            os.makedirs("data/results/plots")
        pacf_figure.savefig(f"data/results/plots/{args.filename}.png")


def lag_plot(
    x: Matrix[Literal["N"], Float],
    lag: int,
    args: PlotOptions,
) -> None:
    x = remove_nans(x)

    nrows = lag // 5
    lag_figure, ax = plt.subplots(nrows, 5, figsize=(20, nrows * 10))
    for i in range(lag):
        x_lag = x[i:]
        ax[i // 5, i % 5].scatter(x[: len(x) - i], x_lag)
        ax[i // 5, i % 5].set_title(f"Lag {i + 1}")
        ax[i // 5, i % 5].set_xlabel(args.x_axis)
        ax[i // 5, i % 5].set_ylabel(args.y_axis)

    lag_figure.suptitle(args.title)
    lag_figure.tight_layout()

    if args.save:
        if not os.path.exists("data/results/plots"):
            os.makedirs("data/results/plots")

        lag_figure.savefig(f"data/results/plots/{args.filename}.png")
