import argparse
import os
from typing import Literal

import pandas as pd

from src.data import clean_dataset, serialize_country_data
from src.models import (
    baum_welch,
    construct_starting_markov_chain,
    prepate_input_for_hmm,
)
from src.structs import Country, Indicator, MarkovChain
from src.utils import Float, Matrix


def _run_baum_welch(country: Country) -> tuple[MarkovChain, MarkovChain]:
    if os.path.exists("data/cleaned/dataset.csv"):
        df = pd.read_csv("data/cleaned/dataset.csv")
    else:
        df = clean_dataset(save_intermediate=True)

    country_data: dict[Indicator, Matrix[Literal["N"], Float]] = {}

    country_data = serialize_country_data(df, country)
    country_data = prepate_input_for_hmm(country_data)

    hidden_markov_chain, known_var_markov_chain = construct_starting_markov_chain(
        country_data
    )
    return baum_welch(hidden_markov_chain, known_var_markov_chain, country_data)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(
        "--countries",
        required=True,
        nargs="+",
        help="The countries to run the Baum-Welch algorithm on. For more information use --help",
    )
    arg_parser.add_argument(
        "--help",
        "-h",
        action="help",
        help=f"Run the Baum-Welch algorithm on the chosen countries. The countries can be chosen from the following list: {', '.join([country.name for country in Country])}",
    )
    args = arg_parser.parse_args()

    countries = [Country[country.upper()] for country in args.countries]
    for country in countries:
        print(f"Running Baum-Welch algorithm for {country.name}...")
        hidden_mc, known_mc = _run_baum_welch(country)

        print("Hidden Markov Chain")
        print(hidden_mc)
        print()
        print("Known Variable Markov Chain")
        print(known_mc)

        print("\n--------------------------------------------------\n")
