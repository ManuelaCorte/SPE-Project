from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src.structs import Indicator, SignificanceResult, StationarityTest
from src.utils import Float, Matrix, PlotOptions


def correlation(
    values: dict[Indicator, Matrix[Literal["N"], Float]],
    plot_args: Optional[PlotOptions] = None,
) -> dict[str, Matrix[Literal["N N"], Float]]:
    """
    Computes the correlation between two time series. The correlations computed
    are the Pearson, Kendall and Spearman correlation coefficients.

    The Pearson correlation coefficient measures the degree of relationship between
    linearly related variables.

    Kendall rank correlation is a non-parametric test that measures the strength of
    dependence between two variables.

    The Spearman rank correlation is a non-parametric test that degree of
    association between two variables. It doesn't carry any assumption but needs
    larger sample size to produce accurate results.

    Parameters:
        variables: The time series to compute the correlation for.
        plot_args: If presents, the correlations matrix are plotted using these
        parameters.

    Returns:
        The correlation matrix for the three correlation coefficients.s
    """
    variables = list(values.values())
    s = variables[0].shape
    for var in variables:
        if len(var.shape) > 1:
            raise TypeError("Variables should be one-dimensional arrays")
        if s != var.shape:
            raise TypeError("Variables should have the same number of observations ")

    variables_matrix: Matrix[Literal["M N"], Float] = np.stack(variables, axis=0)
    pearson: Matrix[Literal["N N"], Float] = np.corrcoef(variables_matrix)

    n = len(values)
    kendall: Matrix[Literal["N N"], Float] = np.eye(n, n)
    spearman: Matrix[Literal["N N"], Float] = np.eye(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            kendall_result: Any = stats.kendalltau(
                variables[i], variables[j], nan_policy="raise"
            )
            kendall[i, j] = kendall_result.statistic
            kendall[j, i] = kendall_result.statistic

            spearman_test: Any = stats.spearmanr(
                variables[i], variables[j], nan_policy="raise", axis=1
            )  # type: ignore
            spearman[i, j] = spearman_test.statistic
            spearman[j, i] = spearman_test.statistic

    if plot_args is not None:
        f, ax = plt.subplots(1, n, figsize=(15, 5))
        # plot correlation as heatmap
        ax[0].imshow(pearson, cmap="coolwarm", interpolation="nearest")
        ax[0].set_title("Pearson correlation")

        ax[1].imshow(kendall, cmap="coolwarm", interpolation="nearest")
        ax[1].set_title("Kendall correlation")

        ax[2].imshow(spearman, cmap="coolwarm", interpolation="nearest")
        ax[2].set_title("Spearman correlation")

        indicators: list[str] = [indicator.name for indicator in values.keys()]
        for i in range(3):
            ax[i].set_xticks(range(n))
            ax[i].set_yticks(range(n))
            ax[i].set_xticklabels(indicators)
            ax[i].set_yticklabels(indicators)

        if plot_args.save:
            f.savefig(f"data/results/{plot_args.filename}")
    plt.show()

    return {"pearson": pearson, "kendall": kendall, "spearman": spearman}


def stationarity(
    x: Matrix[Literal["N"], Float], max_lag: Optional[int] = None
) -> tuple[StationarityTest, StationarityTest]:
    """
    Tests for stationarity of a time series using the following tests:
    - Augmented Dickey-Fuller test: Tests the null hypothesis that a unit root is present in a
    time series characteristic polynomial.
    - Kwiatkowski-Phillips-Schmidt-Shin test: Tests the null hypothesis that the time series is
    stationary around a determinisitc trend, notice that this trend doesn't necessarly means
    stationarity around a constant value.

    Parameters:
        x: The time series to check for stationarity.
        max_lag: The maximum number of lags to include in the test.

    Returns:
        The Augmented Dickey-Fuller and Kwiatkowski-Phillips-Schmidt-Shin test results.
    """
    stat, pvalue, lags, _, critvalues, *_ = adfuller(x, maxlag=max_lag)  # type: ignore
    adf_test: StationarityTest = StationarityTest(
        "Augmented Dickey-Fuller",
        "there is a unit root (one of the roots of the characterstic polynomial is 1) --> the series isn't stationary",
        stat,
        pvalue,
        lags,
        critvalues,
    )

    stat, pvalue, lag, critvalues = kpss(x, "ct")  # type: ignore
    kpss_test: StationarityTest = StationarityTest(
        "Kwiatkowski-Phillips-Schmidt-Shin",
        "The observed time series is stationary around a determinisitc trend",
        stat,
        pvalue,
        lag,
        critvalues,
    )

    return adf_test, kpss_test


def homoscedasticity(
    x: Matrix[Literal["N"], Float]
) -> tuple[SignificanceResult, SignificanceResult]:
    """
    Tests for homoscedasticity in populations using the Bartlett and Levene tests.

    The Bartlett test is used to test the null hypothesis that all input samples are from populations
    with equal variances. The Levene test is similar, but better for non-normal populations.

    Parameters:
        x: The populations to check for homoscedasticity.

    Returns:
        The Bartlett and Levene test results.
    """

    stat, pvalue = stats.bartlett(x)
    bartlett = SignificanceResult(
        "Bartlett", "The variances of the populations are equal", stat, pvalue
    )

    stat, pvalue = stats.levene(x)
    levene = SignificanceResult(
        "Levene", "The variances of the populations are equal", stat, pvalue
    )

    return bartlett, levene


def residuals_autocorrelation(
    residuals: Matrix[Literal["N"], Float], max_lag: int
) -> tuple[SignificanceResult, SignificanceResult, SignificanceResult]:
    """
    Tests for autocorrelation in the residuals using the and the Ljung-Box, Box-Pierce test and
    Durbin Watson test.

    The Ljung-Box test is used to test for the absence of serial correlation in a time series. The
    Box-Pierce test is a simplified version of the Ljung-Box test.

    The Durbin-Watson test is used to detect the presence of autocorrelation at lag 1 in the residuals
    and assumes the errors are the result of a first order autoregression.

    Parameters:
        residuals: The residuals to check for autocorrelation.
        max_lag: The maximum number of lags to include in the test.

    Returns:
        The Ljung-Box, Box-Pierce and Durbin-Watson test results.
    """

    test: pd.DataFrame = acorr_ljungbox(
        residuals, boxpierce=True, lags=max_lag, return_df=True
    )

    ljung = SignificanceResult(
        "Ljung-Box",
        "Any of a group of autocorrelations of a time series is different from zero",
        test["lb_stat"],  # type: ignore
        test["lb_pvalue"].values[1] if test["lb_pvalue"].values.size > 1 else 0,  # type: ignore
    )

    pierce = SignificanceResult(
        "Box-Pierce",
        "The residuals are not autocorrelated / is white noise",
        test["bp_stat"],  # type: ignore
        test["bp_pvalue"],  # type: ignore
    )

    dw_test: float = durbin_watson(residuals)
    dw = SignificanceResult(
        "Durbin-Watson",
        "The residuals are not autocorrelated",
        dw_test,
        0,
    )

    return ljung, pierce, dw


def autocorrelation(
    x: Matrix[Literal["N"], Float], lags: Optional[int] = None
) -> tuple[Matrix[Literal["N"], Float], Matrix[Literal["N"], Float]]:
    """
    Computes the autocorrelation and the partial autocorrelation of a time series.

    Parameters:
        x: The time series to compute the autocorrelation for.
        lags: The number of lags to compute the autocorrelation for.

    Returns:
        The autocorrelation of the time series.
    """
    acf_values: Matrix[Literal["N"], Float] = acf(  # type: ignore
        x, nlags=lags, adjusted=True, fft=True
    )
    pacf_values: Matrix[Literal["N"], Float] = pacf(x, nlags=lags)  # type: ignore

    return acf_values, pacf_values
