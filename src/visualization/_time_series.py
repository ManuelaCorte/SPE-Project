import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.graphics.tsaplots as tsa
from numpy.typing import NDArray


def plot_time_series(
    X: NDArray[np.str_],
    Y: NDArray[np.float32],
    title: str,
    window: int,
    save: bool = False,
) -> None:
    if window % 2 == 0:
        raise ValueError("Window must be odd")

    nan_mask = np.isnan(Y)
    X = X[~nan_mask]
    Y = Y[~nan_mask]

    weights = np.repeat(1.0 / window, window)
    moving_average: NDArray[np.float32] = np.convolve(Y, weights, "valid")

    # pad the moving average with NaNs
    pad = window // 2
    moving_average = np.pad(
        moving_average,
        (pad,),
        mode="constant",
        constant_values=np.NaN,
    )

    sns.set_theme(style="darkgrid")
    sns.lineplot(x=X, y=Y, label="Original")
    sns.lineplot(x=X, y=moving_average, label="Moving Average")
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
    X: NDArray[np.str_],
    Y: NDArray[np.float32],
    lag: int,
    title: str,
    save: bool = False,
) -> None:
    nan_mask = np.isnan(Y)
    X = X[~nan_mask]
    Y = Y[~nan_mask]

    acf_figure, ax = plt.subplots(1, 2, figsize=(20, 10))

    tsa.plot_acf(Y, lags=lag, title=title, ax=ax[0])
    tsa.plot_pacf(Y, lags=lag, title=title, ax=ax[1])

    ax[0].set_xlabel("Lag")
    ax[0].set_ylabel("Autocorrelation")
    ax[1].set_xlabel("Lag")
    ax[1].set_ylabel("Partial Autocorrelation")

    # Lag plot
    nrows = lag // 5 + 1
    lag_figure, ax = plt.subplots(nrows, 5, figsize=(20, nrows * 10))
    for i in range(lag):
        Y_lag = np.pad(Y[i:], (0, i), mode="constant", constant_values=np.NaN)
        ax[i // 5, i % 5].scatter(Y, Y_lag)

    plt.show()

    if save:
        if not os.path.exists("data/results/plots"):
            os.makedirs("data/results/plots")

        acf_figure.savefig(f"data/results/plots/{title}.png")
        lag_figure.savefig(f"data/results/plots/{title}_lag.png")
