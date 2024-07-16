from pprint import pprint
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper

from src.statistics import residuals_autocorrelation
from src.structs import Indicator
from src.utils import Float, Matrix


def _prais_winsten(
    model: RegressionResultsWrapper, rho: float
) -> RegressionResultsWrapper:
    x = model.model.exog
    y = model.model.endog

    # prais winsten transformation for first element
    x_0: Matrix[Literal["1"], Float] = np.sqrt(1 - rho**2) * x[0]
    y_0: Matrix[Literal["1"], Float] = np.sqrt(1 - rho**2) * y[0]

    # cochran orcutt transformation for the rest of the elements
    x_t: Matrix[Literal["N - 1"], Float] = x[1:,] - rho * x[:-1,]
    x_t: Matrix[Literal["N "], Float] = np.append([x_0], x_t, axis=0)
    y_t: Matrix[Literal["N - 1"], Float] = y[1:] - rho * y[:-1]
    y_t: Matrix[Literal["N"], Float] = np.append(y_0, y_t)

    model_ar1 = OLS(y_t, x_t).fit()
    return model_ar1


def prais_winsten_estimation(
    x: Matrix[Literal["N M"], Float],
    y: Matrix[Literal["N"], Float],
    tolerance: float = 1e-3,
) -> RegressionResultsWrapper:
    model = OLS(y, x).fit()
    e_0: Matrix[Literal["N - 1"], Float] = model.resid[1:]
    e_1: Matrix[Literal["N - 1"], Float] = model.resid[:-1]
    rho = np.dot(e_1, e_0) / np.dot(e_1, e_1)
    rho = rho.item()
    model1 = _prais_winsten(model, rho)

    dw = residuals_autocorrelation(model1.resid, 1)[2].statistic
    tolerance = 1e-3
    while dw < 2 - tolerance or dw > 2 + tolerance:
        model1 = _prais_winsten(model1, rho)
        e_0 = model1.resid[1:]
        e_1 = model1.resid[:-1]
        rho = np.dot(e_1, e_0) / np.dot(e_1, e_1)
        rho = rho.item()
        dw = residuals_autocorrelation(model1.resid, 1)[2].statistic
        print("Rho = ", rho)

    return model1


def bootstrap_correlation(
    ts1: Matrix[Literal["N"], Float],
    ts2: Matrix[Literal["N"], Float],
    repetitions: int,
    alpha: float,
) -> list[dict[str, Any]]:
    pearson_correlations: list[float] = []
    kendall_correlations: list[float] = []
    spearman_correlations: list[float] = []

    num_samples = ts1.shape[0]
    for _ in range(repetitions):
        indices = np.random.choice(num_samples, num_samples, replace=True)
        sample1 = ts1[indices]
        sample2 = ts2[indices]
        pearson_correlations.append(np.corrcoef(sample1, sample2)[0, 1])
        kendall_correlations.append(kendalltau(sample1, sample2).statistic)  # type: ignore
        spearman_correlations.append(spearmanr(sample1, sample2).statistic)  # type: ignore

    pearson = {
        "mean": np.mean(pearson_correlations),
        "std": np.std(pearson_correlations),
        "confidence_interval": [
            np.percentile(pearson_correlations, 100 * alpha / 2),
            np.percentile(pearson_correlations, 100 * (1 - alpha / 2)),
        ],
    }

    kendall = {
        "mean": np.mean(kendall_correlations),
        "std": np.std(kendall_correlations),
        "confidence_interval": [
            np.percentile(kendall_correlations, 100 * alpha / 2),
            np.percentile(kendall_correlations, 100 * (1 - alpha / 2)),
        ],
    }

    spearman = {
        "mean": np.mean(spearman_correlations),
        "std": np.std(spearman_correlations),
        "confidence_interval": [
            np.percentile(spearman_correlations, 100 * alpha / 2),
            np.percentile(spearman_correlations, 100 * (1 - alpha / 2)),
        ],
    }
    ("Pearson correlation")
    pprint(pearson)
    pprint("Spearman correlation")
    pprint(spearman)
    pprint("Kendall correlation")
    pprint(kendall)

    return [pearson, kendall, spearman]


def plot_multiple_correlations(features: dict[Indicator, Matrix[Literal["N"], Float]]):
    n = len(features)
    pearson_matrix = np.ones((n, n))
    spearmanr_matrix = np.ones((n, n))
    kendall_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            name1 = Indicator.get_all_names()[i]
            name2 = Indicator.get_all_names()[j]
            indicator1 = Indicator(name1)
            indicator2 = Indicator(name2)
            print(f"Correlation between {indicator1.name} and {indicator2.name}")
            corr = bootstrap_correlation(
                features[indicator1],
                features[indicator2],
                1000,
                0.05,
            )

            pearson_matrix[i, j] = corr[0]["mean"]
            pearson_matrix[j, i] = corr[0]["mean"]
            spearmanr_matrix[i, j] = corr[1]["mean"]
            spearmanr_matrix[j, i] = corr[1]["mean"]
            kendall_matrix[i, j] = corr[2]["mean"]
            kendall_matrix[j, i] = corr[2]["mean"]

    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    cax = ax[0].matshow(pearson_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax[0].set_title("Pearson")
    for i in range(3):
        for j in range(i, 3):
            ax[0].text(
                j,
                i,
                round(pearson_matrix[i, j], 2),
                ha="center",
                va="center",
                color="black",
            )

    cax = ax[1].matshow(kendall_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax[1].set_title("Kendall")
    for i in range(
        3,
    ):
        for j in range(i, 3):
            ax[1].text(
                j,
                i,
                round(kendall_matrix[i, j], 2),
                ha="center",
                va="center",
                color="black",
            )

    cax = ax[2].matshow(spearmanr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax[2].set_title("Spearman")
    for i in range(3):
        for j in range(i, 3):
            ax[2].text(
                j,
                i,
                round(spearmanr_matrix[i, j], 2),
                ha="center",
                va="center",
                color="black",
            )

    for i in range(3):
        ax[i].set_xticks(range(3))
        ax[i].set_yticks(range(3))
        ax[i].set_xticklabels([indicator.name for indicator in Indicator])
        ax[i].set_yticklabels([indicator.name for indicator in Indicator])

    f.colorbar(cax, ax=ax, orientation="horizontal")
    plt.show()
