import argparse
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.data import create_countries_data, divide_training_test_covid_data
from src.models import PraisWinstenRegression
from src.structs import Country
from src.structs._constants import Indicator
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

    y_train = training[Indicator.GDP] / 10e9
    y_test = test[Indicator.GDP] / 10e9
    y_covid = covid[Indicator.GDP] / 10e9
    return ((x_train, y_train), (x_test, y_test), (x_covid, y_covid))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(
        "--country",
        "-c",
        default=DEFAULT_COUNTRY.name,
        help="The country to compute the correlation for.",
    )
    arg_parser.add_argument(
        "--tolerance",
        "-r",
        default=1e-3,
        help="Tolerance level foe the Durbin-Watson test.",
    )
    arg_parser.add_argument(
        "--alpha",
        "-a",
        default=0.05,
        help="Significance level used for computing the bootstrapped confidence interval.",
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
            f"Alpha must be a float between 0 and 1. Instead got: {args.alpha}"
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

    (x, y), (test_x, test_y), (covid_x, covid_y) = prepare_data(
        training, test, covid, args.add_constant
    )

    pw = PraisWinstenRegression(x, y, tolerance)
    pw.fit()
    print(pw.summary())
    pw.plot(f"Diagnostic Plots for {country.name.title()}")
    years = dates[country]
    pw.predict(years, test_x, test_y, covid_x, covid_y)
    plt.show()
