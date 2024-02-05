import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import statsmodels.graphics.tsaplots as tsa

from src.structs import PlotOptions
from src.utils import Float, Matrix, remove_nans


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

    sbn.set_theme(style="darkgrid")
    sbn.lineplot(x=x, y=y, label=args.labels[0])
    sbn.lineplot(x=x, y=moving_average, label=args.labels[1])
    plt.title(args.title)
    plt.xlabel(args.x_axis)
    plt.ylabel(args.y_axis)
    plt.legend()

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
