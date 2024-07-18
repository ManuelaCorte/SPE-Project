from typing import Any, Literal

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.structs import Indicator
from src.utils import Float, Matrix


def bootstrap_correlation(
    features: dict[Indicator, Matrix[Literal["N"], Float]],
    repetitions: int,
    alpha: float,
) -> tuple[
    tuple[Matrix[Literal["N N"], Float], Matrix[Literal["N N 2"], Float]],
    tuple[Matrix[Literal["N N"], Float], Matrix[Literal["N N 2"], Float]],
    tuple[Matrix[Literal["N N"], Float], Matrix[Literal["N N 2"], Float]],
]:
    """
    Compute the mutual correlation between all indicators in the features dictionary.

    Parameters:
        features: A dictionary containing the features to compute the correlation for.
        repetitions: The number of repetitions used for bootstrapping the correlation.
        alpha: The significance level used for computing the bootstrapped confidence interval.

    Returns:
        The Pearson, Spearman and Kendall correlation matrices.
    """
    n = len(features)
    pearson_matrix = np.ones((n, n))
    kendall_matrix = np.ones((n, n))
    spearmanr_matrix = np.ones((n, n))

    pearson_ci_matrix = np.ones((n, n, 2))
    kendall_ci_matrix = np.ones((n, n, 2))
    spearmanr_ci_matrix = np.ones((n, n, 2))

    for i in range(n):
        for j in range(i + 1, n):
            name1 = Indicator.get_all_names()[i]
            name2 = Indicator.get_all_names()[j]
            indicator1 = Indicator(name1)
            indicator2 = Indicator(name2)
            print(f"Correlation between {indicator1.name} and {indicator2.name}")
            corr = _bootstrapping(
                features[indicator1],
                features[indicator2],
                repetitions=repetitions,
                alpha=alpha,
            )

            pearson_matrix[i, j] = corr[0]["mean"]
            pearson_matrix[j, i] = corr[0]["mean"]
            pearson_ci_matrix[i, j] = corr[0]["confidence_interval"]
            pearson_ci_matrix[j, i] = corr[0]["confidence_interval"]

            kendall_matrix[i, j] = corr[2]["mean"]
            kendall_matrix[j, i] = corr[2]["mean"]
            kendall_ci_matrix[i, j] = corr[2]["confidence_interval"]
            kendall_ci_matrix[j, i] = corr[2]["confidence_interval"]

            spearmanr_matrix[i, j] = corr[1]["mean"]
            spearmanr_matrix[j, i] = corr[1]["mean"]
            spearmanr_ci_matrix[i, j] = corr[1]["confidence_interval"]
            spearmanr_ci_matrix[j, i] = corr[1]["confidence_interval"]

    return (
        (pearson_matrix, pearson_ci_matrix),
        (kendall_matrix, kendall_ci_matrix),
        (spearmanr_matrix, spearmanr_ci_matrix),
    )


def _bootstrapping(
    ts1: Matrix[Literal["N"], Float],
    ts2: Matrix[Literal["N"], Float],
    repetitions: int,
    alpha: float,
) -> list[dict[str, Any]]:
    """Computes the Pearson, Kendall and Spearman correlation between two time series using bootstrapping."""
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

    return [pearson, kendall, spearman]
