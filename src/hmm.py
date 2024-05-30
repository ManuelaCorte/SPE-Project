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
from src.structs import Country, Indicator
from src.utils import Float, Matrix

if __name__ == "__main__":
    default_country = Country.UNITED_STATES
    
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(
        "--epochs",
        "-e",
        help="Epochs for training. For more information use --help",
    )
    arg_parser.add_argument(
        "--country",
        "-c",
        help="The countries to run the Baum-Welch algorithm on. For more information use --help",
    )
    arg_parser.add_argument(
        "--help",
        "-h",
        action="help",
        help=f"Run the Baum-Welch algorithm. The starting country can be chosen from the following list (DEFAULT: {default_country.name}): {', '.join([country.name for country in Country])}",
    )
    args = arg_parser.parse_args()

    print("Hidden Markov Chain States:\n" + 
          "\t0: increase Interest Rate, increase Consumer Price Index\n" + 
          "\t1: increase Interest Rate, decrease Consumer Price Index\n" + 
          "\t2: decrease Interest Rate, increase Consumer Price Index\n" + 
          "\t3: decrease Interest Rate, decrease Consumer Price Index\n")
    print("Known Variable Markov Chain States:\n" + 
          "\t0: increase Gross Domestic Product\n" + 
          "\t1: decrease Gross Domestic Product\n")

    epochs = int(args.epochs) if args.epochs and args.epochs.isdigit() else 25
    starting_country = (
        Country[args.country.upper()]
        if args.country
        and args.country.upper() in [country.name for country in Country]
        else default_country
    )
    print(f"Starting country: {starting_country.name}")
    print(f"Epochs: {epochs}")

    countries = Country
    countries_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]] = {}
    for country in countries:
        if os.path.exists("data/cleaned/dataset.csv"):
            df = pd.read_csv("data/cleaned/dataset.csv")
        else:
            df = clean_dataset(save_intermediate=True)

        country_data: dict[Indicator, Matrix[Literal["N"], Float]] = {}

        country_data = serialize_country_data(df, country)
        country_data = prepate_input_for_hmm(country_data)
        countries_data[country] = country_data

    hidden_markov_chain, known_var_markov_chain = construct_starting_markov_chain(
        countries_data[starting_country]
    )

    print("\n--------------------------------------------------\n")
    print("Hidden Markov Chain")
    print(hidden_markov_chain)
    print()
    print("Known Variable Markov Chain")
    print(known_var_markov_chain)
    print("\n--------------------------------------------------\n")

    hidden_mc, known_mc = baum_welch(
        hidden_markov_chain, known_var_markov_chain, countries_data, epochs
    )

    print("\n--------------------------------------------------\n")
    print("Hidden Markov Chain")
    print(hidden_markov_chain)
    print()
    print("Known Variable Markov Chain")
    print(known_var_markov_chain)
    print("\n--------------------------------------------------\n")

    filename = "hmm"
    hidden_markov_chain.to_image_with_known_var(filename, known_var_markov_chain)
    print(f"Graph saved to data/results/{filename}.png")
