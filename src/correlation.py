import argparse
from pprint import pprint
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.data import create_countries_data
from src.models import bootstrap_correlation
from src.structs import Country, Indicator
from src.utils import Float, Matrix

DEFAULT_COUNTRY = Country.ITALY


def plot_multiple_correlations(
    pearson: tuple[Matrix[Literal["N N"], Float], Matrix[Literal["N N 2"], Float]],
    kendall: tuple[Matrix[Literal["N N"], Float], Matrix[Literal["N N 2"], Float]],
    spearmanr: tuple[Matrix[Literal["N N"], Float], Matrix[Literal["N N 2"], Float]],
    country: Optional[Country] = None,
) -> None:
    lower = _plot_correlation(
        pearson[1][:, :, 0], kendall[1][:, :, 0], spearmanr[1][:, :, 0]
    )
    lower.suptitle(
        "Lower bound of the confidence interval for the correlation", fontsize=16
    )

    mean = _plot_correlation(pearson[0], kendall[0], spearmanr[0])
    mean.suptitle("Mean of the correlation", fontsize=16)

    upper = _plot_correlation(
        pearson[1][:, :, 1], kendall[1][:, :, 1], spearmanr[1][:, :, 1]
    )
    upper.suptitle(
        "Upper bound of the confidence interval for the correlation", fontsize=16
    )

    print("Saving plots to data/results/")
    if country:
        lower.savefig(f"data/results/{country.name}_lower.png")
        mean.savefig(f"data/results/{country.name}_mean.png")
        upper.savefig(f"data/results/{country.name}_upper.png")
    else:
        lower.savefig("data/results/all_countries_lower.png")
        mean.savefig("data/results/all_countries_mean.png")
        upper.savefig("data/results/all_countries_upper.png")
    plt.plot()


def _plot_correlation(
    pearson_matrix: Matrix[Literal["N N"], Float],
    kendall_matrix: Matrix[Literal["N N"], Float],
    spearmanr_matrix: Matrix[Literal["N N"], Float],
) -> Figure:
    n = pearson_matrix.shape[0]
    f, ax = plt.subplots(1, n, figsize=(15, 7))
    cax = ax[0].matshow(pearson_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax[0].set_title("Pearson")
    for i in range(n):
        for j in range(i, n):
            ax[0].text(
                j,
                i,
                round(pearson_matrix[i, j], 4),
                ha="center",
                va="center",
                color="black",
            )

    cax = ax[1].matshow(kendall_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax[1].set_title("Kendall")
    for i in range(n):
        for j in range(i, n):
            ax[1].text(
                j,
                i,
                round(kendall_matrix[i, j], 4),
                ha="center",
                va="center",
                color="black",
            )

    cax = ax[2].matshow(spearmanr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax[2].set_title("Spearman")
    for i in range(n):
        for j in range(i, n):
            ax[2].text(
                j,
                i,
                round(spearmanr_matrix[i, j], 4),
                ha="center",
                va="center",
                color="black",
            )

    for i in range(n):
        ax[i].set_xticks(range(n))
        ax[i].set_yticks(range(n))
        ax[i].set_xticklabels([indicator.name for indicator in Indicator])
        ax[i].set_yticklabels([indicator.name for indicator in Indicator])

    f.colorbar(cax, ax=ax, orientation="horizontal")
    return f


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(
        "--repetitions",
        "-r",
        default=1000,
        help="Number of repetitions used for bootstrapping the correlation.",
    )
    arg_parser.add_argument(
        "--alpha",
        "-a",
        default=0.05,
        help="Significance level used for computing the bootstrapped confidence interval.",
    )
    arg_parser.add_argument(
        "--country",
        "-c",
        default=DEFAULT_COUNTRY.name,
        help="The country to compute the correlation for.",
    )
    arg_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Compute overall correlation for all countries.",
    )
    arg_parser.add_argument(
        "--help",
        "-h",
        action="help",
        help=f"""Compute correlation between GDP, Interest Rates and Consumer Price Index for a country.
        The country can be chosen from the following list (DEFAULT: {DEFAULT_COUNTRY.name}): {', '.join([country.name for country in Country])}""",
    )
    args = arg_parser.parse_args()

    if int(args.repetitions) <= 0:
        raise ValueError("Repetitions must be a positive integer.")
    repetitions = int(args.repetitions)

    if not 0 < float(args.alpha) < 1:
        raise ValueError(
            f"Alpha must be a float between 0 and 1. Instead got: {args.alpha}"
        )
    alpha = float(args.alpha)

    country = Country[args.country.upper()]
    if country.name not in [country.name for country in Country]:
        raise ValueError(
            f"Country must be one of the following: {', '.join([country.name for country in Country])}"
        )

    pearson, spearmanr, kendall = None, None, None
    if args.all:
        features: dict[Indicator, Matrix[Literal["N"], Float]] = {}
        countries, countries_data, dates = create_countries_data(
            country, all_countries=True, pct=False
        )
        print("Computing correlation for all countries")
        for data in countries_data.values():
            for indicator, matrix in data.items():
                if indicator not in features:
                    features[indicator] = matrix
                else:
                    features[indicator] = np.concatenate(
                        (features[indicator], matrix), axis=0
                    )
        pearson, spearmanr, kendall = bootstrap_correlation(
            features, repetitions, alpha
        )
    else:
        countries, countries_data, dates = create_countries_data(
            country, all_countries=False, pct=False
        )
        print(f"Computing correlation for {country.name}")
        features = countries_data[country]
        pearson, spearmanr, kendall = bootstrap_correlation(
            features, repetitions, alpha
        )
    print("Pearson correlation matrix:")
    pprint(pearson[0])
    print("Pearson confidence interval:")
    pprint(pearson[1])
    print("Spearman correlation matrix:")
    pprint(spearmanr[0])
    print("Spearman confidence interval:")
    pprint(spearmanr[1])
    print("Kendall correlation matrix:")
    pprint(kendall[0])

    plot_multiple_correlations(
        pearson, kendall, spearmanr, country=None if args.all else country
    )
