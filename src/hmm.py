import argparse
import os
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import clean_dataset, serialize_country_data
from src.models import baum_welch, construct_starting_markov_chain
from src.structs import Country, HiddenState, Indicator, KnownVariables, MarkovChain
from src.utils import Float, Matrix


def create_countries_data(starting_country: Country, multiple_series: bool = False):
    countries = Country.get_all_countries() if multiple_series else [starting_country]
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

    print()
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

        training_points = len(training_data[country][Indicator.GDP])
        testing_points = len(test_data[country][Indicator.GDP])
        covid_points = len(covid_data[country][Indicator.GDP])

        country_name = "{:<13}".format(country.name)
        print(
            f"Country {country_name} starts from {this_date[0]} and ends at {this_date[-1]} ({training_points} training, {testing_points} test, {covid_points} covid)"
        )
    print()

    return training_data, test_data, covid_data


def create_countries_data_divided(
    starting_country: Country, multiple_series: bool = False
):
    countries, countries_data, dates = create_countries_data(
        starting_country, multiple_series
    )
    training_data, test_data, covid_data = divide_training_test_covid_data(
        countries, countries_data, dates
    )
    return countries, training_data, test_data, covid_data


def test_markov_chain(
    starting_country: Country,
    test_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]],
    last_state: HiddenState,
    hidden_markov_chain: MarkovChain,
    known_var_markov_chain: MarkovChain,
):
    total_testing_data = len(test_data[starting_country][Indicator.GDP])

    positive_testing_data = 0
    negative_testing_data = 0
    for value in test_data[starting_country][Indicator.GDP]:
        if value > 0:
            positive_testing_data += 1
        else:
            negative_testing_data += 1
    print(
        f"{starting_country.name} testing data has {positive_testing_data} positive and {negative_testing_data} negative values for GDP"
    )

    epochs = 10000
    hidden_states = np.zeros((len(HiddenState)))
    known_states = np.zeros((len(KnownVariables)))
    curr_state = last_state.value
    for _ in tqdm(range(epochs), desc="testing", unit="epoch"):
        for _ in range(total_testing_data):
            curr_state = hidden_markov_chain.random_walk(curr_state)
            known_var_state = known_var_markov_chain.random_walk(curr_state)
            hidden_states[curr_state] += 1
            known_states[known_var_state] += 1
    print(
        f"{starting_country.name} random walks has produced on average {hidden_states / epochs} for the hidden states and {known_states / epochs} for the known states"
    )


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
        "--multiple-series",
        "-m",
        action="store_true",
        help="You can train the Markov chain on all countries series instead of just the starting country",
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
    multiple_series = args.multiple_series
    print(f"Starting country: {starting_country.name}")
    print(f"Epochs: {epochs}")
    print(f"Use multiple series: {multiple_series}")

    countries, training_data, test_data, covid_data = create_countries_data_divided(
        starting_country, multiple_series
    )

    hidden_markov_chain, known_var_markov_chain = construct_starting_markov_chain(
        training_data[starting_country]
    )

    print("\n--------------------------------------------------\n")
    print(f"Initial chain created from training data of {starting_country.name}")
    print()
    print("Hidden Markov Chain")
    print(hidden_markov_chain)
    print()
    print("Known Variable Markov Chain")
    print(known_var_markov_chain)
    print("\n--------------------------------------------------\n")

    baum_welch(
        hidden_markov_chain, known_var_markov_chain, training_data, countries, epochs
    )

    print("\n--------------------------------------------------\n")
    print("Final chain after training")
    print()
    print("Hidden Markov Chain")
    print(hidden_markov_chain)
    print()
    print("Known Variable Markov Chain")
    print(known_var_markov_chain)
    print("\n--------------------------------------------------\n")

    filename = "hmm"
    hidden_markov_chain.to_image_with_known_var(filename, known_var_markov_chain)
    print(f"Graph saved to data/results/{filename}.png")
    print("\n--------------------------------------------------\n")

    print("Use test data")
    last_state: HiddenState = HiddenState.get_state(
        training_data[starting_country][Indicator.IR][-1],
        training_data[starting_country][Indicator.CPI][-1],
    )
    test_markov_chain(
        starting_country,
        test_data,
        last_state,
        hidden_markov_chain,
        known_var_markov_chain,
    )
    print()
    print("Use covid data")
    last_state: HiddenState = HiddenState.get_state(
        test_data[starting_country][Indicator.IR][-1],
        test_data[starting_country][Indicator.CPI][-1],
    )
    test_markov_chain(
        starting_country,
        covid_data,
        last_state,
        hidden_markov_chain,
        known_var_markov_chain,
    )
