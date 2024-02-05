import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.graphics.tsaplots as tsa

from src.utils import Matrix


def plot_time_series(
    x: Matrix[Literal["N"], np.str_],
    y: Matrix[Literal["N"], np.float32],
    title: str,
    window: int,
    save: bool = False,
) -> None:
    if window % 2 == 0:
        raise ValueError("Window must be odd")

    nan_mask = np.isnan(y)
    x = x[~nan_mask]
    y = y[~nan_mask]

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

    sns.set_theme(style="darkgrid")
    sns.lineplot(x=x, y=y, label="Original")
    sns.lineplot(x=x, y=moving_average, label="Moving Average")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    if save:
        if not os.path.exists("data/results/plots"):
            os.makedirs("data/results/plots")
        plt.savefig(f"plots/{title}.png")


def autocorrelation_plots(
    x: Matrix[Literal["N"], np.str_],
    y: Matrix[Literal["N"], np.float32],
    lag: int,
    title: str,
    save: bool = False,
) -> None:
    nan_mask = np.isnan(y)
    x = x[~nan_mask]
    y = y[~nan_mask]

    acf_figure, ax = plt.subplots(1, 2, figsize=(20, 10))

    tsa.plot_acf(y, lags=lag, title=title, ax=ax[0])
    tsa.plot_pacf(y, lags=lag, title=title, ax=ax[1])

    ax[0].set_xlabel("Lag")
    ax[0].set_ylabel("Autocorrelation")
    ax[1].set_xlabel("Lag")
    ax[1].set_ylabel("Partial Autocorrelation")

    # Lag plot
    nrows = lag // 5
    lag_figure, ax = plt.subplots(nrows, 5, figsize=(20, nrows * 10))
    for i in range(lag):
        Y_lag = np.roll(y, i + 1)
        ax[i // 5, i % 5].scatter(y, Y_lag)
        ax[i // 5, i % 5].set_title(f"Lag {i + 1}")

    plt.show()

    if save:
        if not os.path.exists("data/results/plots"):
            os.makedirs("data/results/plots")

        acf_figure.savefig(f"data/results/plots/{title}.png")
        lag_figure.savefig(f"data/results/plots/{title}_lag.png")
