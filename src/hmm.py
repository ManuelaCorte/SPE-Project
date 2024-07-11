import argparse
import os
from typing import Literal

import numpy as np
import pandas as pd

from src.data import clean_dataset, serialize_country_data
from src.models import baum_welch, construct_starting_markov_chain
from src.structs import Country, Indicator
from src.utils import Float, Matrix


def create_countries_data(starting_country: Country, multi_series: bool = False):
    countries = Country.get_all_countries() if multi_series else [starting_country]
    countries_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]] = {}
    dates: dict[Country, Matrix[Literal["N"], np.str_]] = {}
    for country in countries:
        if os.path.exists("data/cleaned/dataset.csv"):
            df = pd.read_csv("data/cleaned/dataset.csv")
        else:
            df = clean_dataset(save_intermediate=True)

        country_data: dict[Indicator, Matrix[Literal["N"], Float]] = {}

        country_data, date = serialize_country_data(df, country, pct=True)
        dates[country] = date
        # country_data = prepate_input_for_hmm(country_data)
        countries_data[country] = country_data
    return countries, countries_data, dates


def divide_training_test_covid_data(
    countries: list[Country],
    countries_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]],
    dates: dict[Country, Matrix[Literal["N"], np.str_]],
):
    TESTING_YEAR = "2018"
    COVID_YEAR = "2020"
    print(
        f"Data is divided between training, testing (after {TESTING_YEAR}) and flawed due to covid (after {COVID_YEAR})"
    )

    training_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]] = {}
    test_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]] = {}
    covid_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]] = {}
    for country in countries:
        country_data = countries_data[country]

        training_data[country] = {}
        test_data[country] = {}
        covid_data[country] = {}

        this_date = dates[country]
        test_date_index = -1
        covid_date_index = -1
        for i in range(len(this_date)):
            if test_date_index < 0 and TESTING_YEAR in this_date[i]:
                test_date_index = i
            if covid_date_index < 0 and COVID_YEAR in this_date[i]:
                covid_date_index = i

        for indicator in country_data:
            data = country_data[indicator]
            training_data[country][indicator] = data[:test_date_index:]
            test_data[country][indicator] = data[test_date_index:covid_date_index:]
            covid_data[country][indicator] = data[covid_date_index::]

    return training_data, test_data, covid_data


def create_countries_data_divided(
    starting_country: Country, multi_series: bool = False
):
    countries, countries_data, dates = create_countries_data(
        starting_country, multi_series
    )
    training_data, test_data, covid_data = divide_training_test_covid_data(
        countries, countries_data, dates
    )
    return countries, training_data, test_data, covid_data


if __name__ == "__main__":
    DEFAULT_COUNTRY = Country.ITALY
    DEFAULT_EPOCHS = 3

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
        help=f"Run the Baum-Welch algorithm. The starting country can be chosen from the following list (DEFAULT: {DEFAULT_COUNTRY.name}): {', '.join([country.name for country in Country])}",
    )
    args = arg_parser.parse_args()

    print(
        "Hidden Markov Chain States:\n"
        + "\t0: increase Interest Rate, increase Consumer Price Index\n"
        + "\t1: increase Interest Rate, decrease Consumer Price Index\n"
        + "\t2: decrease Interest Rate, increase Consumer Price Index\n"
        + "\t3: decrease Interest Rate, decrease Consumer Price Index\n"
    )
    print(
        "Known Variable Markov Chain States:\n"
        + "\t0: increase Gross Domestic Product\n"
        + "\t1: decrease Gross Domestic Product\n"
    )

    epochs = (
        int(args.epochs) if args.epochs and args.epochs.isdigit() else DEFAULT_EPOCHS
    )
    starting_country = (
        Country[args.country.upper()]
        if args.country
        and args.country.upper() in [country.name for country in Country]
        else DEFAULT_COUNTRY
    )
    print(f"Starting country: {starting_country.name}")
    print(f"Epochs: {epochs}")

    countries, training_data, test_data, covid_data = create_countries_data_divided(
        starting_country
    )

    hidden_markov_chain, known_var_markov_chain = construct_starting_markov_chain(
        training_data[starting_country]
    )

    print("\n--------------------------------------------------\n")
    print("Hidden Markov Chain")
    print(hidden_markov_chain)
    print()
    print("Known Variable Markov Chain")
    print(known_var_markov_chain)
    print("\n--------------------------------------------------\n")

    hidden_mc, known_mc = baum_welch(
        hidden_markov_chain, known_var_markov_chain, training_data, countries, epochs
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
