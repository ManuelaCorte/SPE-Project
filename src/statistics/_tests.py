from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.figure import Figure
from statsmodels.graphics.correlation import plot_corr_grid
from statsmodels.tsa.stattools import adfuller, kpss

from src.structs import PlotOptions, SignificanceResult, StationarityTest
from src.utils import Float, Matrix


def correlation(
    variables: list[Matrix[Literal["N"], Float]],
    plot_args: Optional[PlotOptions] = None,
) -> dict[str, Matrix[Literal["N N"], Float]]:
    """
    Computes the correlation between two time series. The correlations computed
    are the Pearson, Kendall and Spearman correlation coefficients.
    """
    s = variables[0].shape
    for var in variables:
        if len(var.shape) > 1:
            raise TypeError("Variables should be one-dimensional arrays")
        if s != var.shape:
            raise TypeError("Variables should have the same number of observations ")

    variables_matrix: Matrix[Literal["M N"], Float] = np.stack(variables, axis=0)
    pearson: Matrix[Literal["N N"], Float] = np.corrcoef(variables_matrix)

    m = len(variables)
    kendall: Matrix[Literal["N N"], Float] = np.zeros((m, m))
    spearman: Matrix[Literal["N N"], Float] = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            kendall_result: SignificanceResult = stats.kendalltau(
                variables[i], variables[j], nan_policy="raise"
            )
            kendall[i, j] = kendall_result.statistic
            kendall[j, i] = kendall_result.statistic

            spearman_test: SignificanceResult = stats.spearmanr(
                variables[i], variables[j], nan_policy="raise", axis=1
            )  # type: ignore
            spearman[i, j] = spearman_test.statistic
            spearman[j, i] = spearman_test.statistic

    if plot_args is not None:
        f: Figure = plot_corr_grid([pearson, kendall, spearman])
        if plot_args.save:
            f.savefig(f"data/results/{plot_args.filename}")
    plt.show()

    return {"pearson": pearson, "kendall": kendall, "spearman": spearman}


def stationarity(
    x: Matrix[Literal["N"], Float], max_lag: Optional[int] = None
) -> tuple[StationarityTest, StationarityTest]:
    stat, pvalue, lags, _, critvalues, *_ = adfuller(x, maxlag=max_lag)  # type: ignore
    adf_test: StationarityTest = StationarityTest(
        "Augmented Dickey-Fuller", stat, pvalue, lags, critvalues
    )

    stat, pvalue, lag, critvalues = kpss(x)  # type: ignore
    kpss_test: StationarityTest = StationarityTest(
        "Kwiatkowski-Phillips-Schmidt-Shin", stat, pvalue, lag, critvalues
    )

    return adf_test, kpss_test
