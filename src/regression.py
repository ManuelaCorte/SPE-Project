import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.data import create_countries_data, divide_training_test_covid_data
from src.models import PraisWinstenRegression
from src.structs import Country
from src.structs._constants import Indicator

DEFAULT_COUNTRY = Country.ITALY

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

    x = np.column_stack([training[Indicator.IR], training[Indicator.CPI]])
    if args.add_constant:
        x = np.column_stack((np.ones(x.shape[0]), x))
    y = training[Indicator.GDP] / 10e9
    pw = PraisWinstenRegression(x, y, tolerance)
    pw.fit()

    pw.plot()
    plt.show()
