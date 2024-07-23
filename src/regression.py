import argparse
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.data import create_countries_data, divide_training_test_covid_data
from src.models import PraisWinstenRegression
from src.statistics import differencing
from src.structs import Country, Indicator
from src.utils import Float, Matrix

DEFAULT_COUNTRY = Country.ITALY


def prepare_data(
    training: dict[Indicator, Matrix[Literal["N"], Float]],
    test: dict[Indicator, Matrix[Literal["N"], Float]],
    covid: dict[Indicator, Matrix[Literal["N"], Float]],
    add_constant: bool,
) -> tuple[
    tuple[Matrix[Literal["M N"], Float], Matrix[Literal["M N"], Float]],
    tuple[Matrix[Literal["M N"], Float], Matrix[Literal["M N"], Float]],
    tuple[Matrix[Literal["M N"], Float], Matrix[Literal["M N"], Float]],
]:
    x_train = np.column_stack([training[Indicator.IR], training[Indicator.CPI]])
    x_test = np.column_stack([test[Indicator.IR], test[Indicator.CPI]])
    x_covid = np.column_stack([covid[Indicator.IR], covid[Indicator.CPI]])
    if add_constant:
        x_train = np.column_stack((np.ones(x_train.shape[0]), x_train))
        x_test = np.column_stack((np.ones(x_test.shape[0]), x_test))
        x_covid = np.column_stack((np.ones(x_covid.shape[0]), x_covid))

    y_train = training[Indicator.GDP] / 1e11
    y_test = test[Indicator.GDP] / 1e11
    y_covid = covid[Indicator.GDP] / 1e11
    return ((x_train, y_train), (x_test, y_test), (x_covid, y_covid))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(
        "--country",
        "-c",
        default=DEFAULT_COUNTRY.name,
        help="The country to compute the regression for.",
    )
    arg_parser.add_argument(
        "--tolerance",
        "-r",
        default=1e-3,
        help="Tolerance level (pvalue) for the Ljung-Box test used to determine convergence.",
    )
    arg_parser.add_argument(
        "--add_constant",
        action="store_true",
        default=True,
        help="Add a constant to the model.",
    )
    arg_parser.add_argument(
        "--help",
        "-h",
        action="help",
        help=f"""The country can be chosen from the following list (DEFAULT: {DEFAULT_COUNTRY.name}): {', '.join([country.name for country in Country])}""",
    )
    args = arg_parser.parse_args()

    if not 0 < float(args.tolerance) < 1:
        raise ValueError(
            f"Tolerance must be a float between 0 and 1. Instead got: {args.alpha}"
        )
    tolerance = float(args.tolerance)

    country = Country[args.country.upper()]
    if country.name not in [country.name for country in Country]:
        raise ValueError(
            f"Country must be one of the following: {', '.join([country.name for country in Country])}"
        )

    pearson, spearmanr, kendall = None, None, None

    countries, countries_data, dates = create_countries_data(
        country, all_countries=False, pct=False
    )
    data = divide_training_test_covid_data(countries, countries_data, dates)
    training = data[0][country]
    test = data[1][country]
    covid = data[2][country]

    if len(test[Indicator.GDP]) == 0 or len(covid[Indicator.GDP]) == 0:
        print(
            f"ERROR: Country {country.name} does not have enough data to perform the regression."
        )
        exit(1)

    (x, y), (test_x, test_y), (covid_x, covid_y) = prepare_data(
        training, test, covid, args.add_constant
    )

    for indicator in [Indicator.GDP, Indicator.IR, Indicator.CPI]:
        training[indicator] = differencing(training[indicator], 1)
        test[indicator] = differencing(test[indicator], 1)
        covid[indicator] = differencing(covid[indicator], 1)
    (
        (x_diff, y_diff),
        (test_x_diff, test_y_diff),
        (covid_x_diff, covid_y_diff),
    ) = prepare_data(training, test, covid, args.add_constant)

    pw = PraisWinstenRegression(x, y, x_diff, y_diff, tolerance)
    pw.fit()
    print(pw.summary())
    diagnostic_figure = pw.diagnostic_plots(
        f"Diagnostic Plots for {country.name.title()}"
    )
    diagnostic_figure.savefig(f"data/results/{country.name}_diagnostic_plots.png")
    years = dates[country]
    fig_diff, fig_one_ahead, fig_predictions = pw.predict(
        years,
        test_x_diff,
        test_y,
        test_y_diff,
        covid_x_diff,
        covid_y,
        covid_y_diff,
    )
    fig_diff.savefig(f"data/results/{country.name}_reg_diff.png")
    fig_one_ahead.savefig(f"data/results/{country.name}_reg_one_ahead.png")
    fig_predictions.savefig(f"data/results/{country.name}_reg_predictions.png")

    plt.show()
