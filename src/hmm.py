import argparse
from typing import Literal

import numpy as np
from scipy import stats
from tqdm import tqdm

from src.data import create_countries_data, divide_training_test_covid_data
from src.models import baum_welch, construct_starting_markov_chain
from src.structs import Country, HiddenState, Indicator, KnownVariables, MarkovChain
from src.utils import Float, Matrix


def create_countries_data_divided(
    starting_country: Country, multiple_series: bool = False
):
    countries, countries_data, dates = create_countries_data(
        starting_country, multiple_series, pct=True
    )
    training_data, test_data, covid_data = divide_training_test_covid_data(
        countries, countries_data, dates
    )
    return countries, training_data, test_data, covid_data


def count_data_instances(
    data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]], name: str
):
    positive_testing_data = 0
    negative_testing_data = 0
    hidden_states_test = np.zeros((len(HiddenState)))
    for i in range(len(data[starting_country][Indicator.GDP])):
        value = data[starting_country][Indicator.GDP][i]
        if value > 0:
            positive_testing_data += 1
        else:
            negative_testing_data += 1

        ir = data[starting_country][Indicator.IR][i]
        cpi = data[starting_country][Indicator.CPI][i]
        state = HiddenState.get_state(ir, cpi)
        hidden_states_test[state.value] += 1

    print(
        f"{starting_country.name} {name} data has {positive_testing_data} positive and {negative_testing_data} negative values for GDP ({hidden_states_test})"
    )


def test_markov_chain(
    starting_country: Country,
    test_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]],
    last_state: HiddenState,
    hidden_markov_chain: MarkovChain,
    known_var_markov_chain: MarkovChain,
):
    epochs = 1000
    total_testing_data = len(test_data[starting_country][Indicator.GDP])
    hidden_states_runs = np.zeros((epochs, len(HiddenState)))
    known_var_runs = np.zeros((epochs, len(KnownVariables)))
    curr_state = last_state.value
    for epoch in tqdm(range(epochs), desc="testing", unit="epoch"):
        hidden_states = np.zeros((len(HiddenState)))
        known_var = np.zeros((len(KnownVariables)))
        for _ in range(total_testing_data):
            curr_state = hidden_markov_chain.random_walk(curr_state)
            known_var_state = known_var_markov_chain.random_walk(curr_state)
            hidden_states[curr_state] += 1
            known_var[known_var_state] += 1
        hidden_states_runs[epoch] = hidden_states
        known_var_runs[epoch] = known_var

    hidden_states_avg = np.mean(hidden_states_runs, axis=0)
    known_var_avg = np.mean(known_var_runs, axis=0)
    hidden_states_std = np.std(hidden_states_runs, axis=0)
    known_var_std = np.std(known_var_runs, axis=0)
    hidden_state_ci = (
        stats.t.ppf(1 - 0.05 / 2, epochs - 1)
        * hidden_states_std
        * np.sqrt(1 + 1 / epochs)
    )
    known_var_ci = (
        stats.t.ppf(1 - 0.05 / 2, epochs - 1) * known_var_std * np.sqrt(1 + 1 / epochs)
    )

    hidden_state_estimate = []
    known_var_estimate = []
    for mean, ci in zip(hidden_states_avg, hidden_state_ci):
        s = f"{mean:.2f} ± {ci:.2f}"
        hidden_state_estimate.append(s)
    for mean, ci in zip(known_var_avg, known_var_ci):
        s = f"{mean:.2f} ± {ci:.2f}"
        known_var_estimate.append(s)

    print(
        f"{starting_country.name} random walks has produced on average {hidden_state_estimate} for the hidden states and {known_var_estimate} for the known states"
    )
    print(
        f"On average we have predicted {known_var_estimate[0]} positive and {known_var_estimate[1]} negative values for GDP"
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

    print(f"Number of countries used: {len(training_data.keys())}")

    if (
        len(test_data[starting_country][Indicator.GDP]) == 0
        or len(covid_data[starting_country][Indicator.GDP]) == 0
    ):
        print(
            f"ERROR: Country {starting_country.name} does not have enough data to test the Baum-Welch algorithm. Use another starting country."
        )
        exit(1)

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

    print("See training data")
    count_data_instances(training_data, "training")
    print()
    print("Use test data")
    last_state: HiddenState = HiddenState.get_state(
        training_data[starting_country][Indicator.IR][-1],
        training_data[starting_country][Indicator.CPI][-1],
    )
    count_data_instances(test_data, "testing")
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
    count_data_instances(covid_data, "covid")
    test_markov_chain(
        starting_country,
        covid_data,
        last_state,
        hidden_markov_chain,
        known_var_markov_chain,
    )
