from copy import deepcopy
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.structs import Country, Indicator, MarkovChain
from src.utils import Float, Matrix

GDP, IR, CPI = Indicator


class HiddenState(Enum):
    """
    The hidden state of the markov chain. Each state tells if the Interest Rate and the CPI are increasing or decreasing
    """

    I_IR_I_CPI = 0
    I_IR_D_CPI = 1
    D_IR_I_CPI = 2
    D_IR_D_CPI = 3

    @staticmethod
    def get_all_states() -> list[int]:
        return [state.value for state in HiddenState]

    @staticmethod
    def get_state(ir: float, cpi: float):
        if ir > 0 and cpi > 0:
            return HiddenState.I_IR_I_CPI
        elif ir > 0 and cpi < 0:
            return HiddenState.I_IR_D_CPI
        elif ir < 0 and cpi > 0:
            return HiddenState.D_IR_I_CPI
        else:
            return HiddenState.D_IR_D_CPI


class KnownVariables(Enum):
    """
    The known variables of the markov chain. Each state tells if the GDP is increasing or decreasing
    """

    I_GDP = 0
    D_GDP = 1

    @staticmethod
    def get_all_variables() -> list[int]:
        return [var.value for var in KnownVariables]

    @staticmethod
    def get_variable(gdp: float):
        return KnownVariables.I_GDP if gdp > 0 else KnownVariables.D_GDP


def baum_welch(
    hidden_markov_chain: MarkovChain,
    known_var_markov_chain: MarkovChain,
    countries_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]],
    epochs: int = 1,
) -> tuple[MarkovChain, MarkovChain]:
    """
    Baum-Welch algorithm for Hidden Markov Models.

    Parameters:
        hidden_markov_chain: The hidden markov chain.
        known_var_markov_chain: The known variables markov chain.
        country_data: The country data.
        epochs: The number of epochs to train the model.

    Returns:
        The trained hidden markov chain and known variables markov chain.
    """
    countries = Country
    for _ in tqdm(range(epochs), desc="training", unit="epoch"):
        A = deepcopy(hidden_markov_chain.transitions)
        B = deepcopy(known_var_markov_chain.transitions)
        pi = deepcopy(hidden_markov_chain.states)
        n = len(hidden_markov_chain.states)

        gammas: list[npt.NDArray[np.float64]] = []
        xis: list[npt.NDArray[np.float64]] = []
        Ys: list[list[int]] = []
        Ts: list[int] = []
        R = len(countries)

        for country in countries:
            country_data = countries_data[country]
            Y = [KnownVariables.get_variable(gdp).value for gdp in country_data[GDP]]
            T = len(Y)
            Ys.append(Y)
            Ts.append(T)

            alpha = _forward(pi, A, B, Y)
            beta = _backward(A, B, Y)

            gamma, xi = _compute_temporary_variables(alpha, beta, A, B, Y)
            gammas.append(gamma)
            xis.append(xi)

        for i in range(n):
            pi[i] = np.sum([gammas[r][i][0] for r in range(R)]) / R

        for i in range(n):
            for j in range(n):
                A[i][j] = np.sum(
                    [xis[r][i][j][t] for r in range(R) for t in range(Ts[r] - 1)]
                ) / np.sum(
                    [gammas[r][i][t] for r in range(R) for t in range(Ts[r] - 1)]
                )

        for i in range(n):
            for j in range(len(KnownVariables.get_all_variables())):
                B[i][j] = np.sum(
                    [
                        gammas[r][i][t]
                        for r in range(R)
                        for t in range(Ts[r])
                        if KnownVariables.get_variable(Ys[r][t]).value == j
                    ]
                ) / np.sum([gammas[r][i][t] for r in range(R) for t in range(Ts[r])])

        hidden_markov_chain.transitions = A
        known_var_markov_chain.transitions = B
        hidden_markov_chain.states = pi

    return hidden_markov_chain, known_var_markov_chain


def prepate_input_for_hmm(
    country_data: dict[Indicator, Matrix[Literal["N"], Float]]
) -> dict[Indicator, Matrix[Literal["N"], Float]]:
    # * take only the difference between a member and its next for each indicator
    length = len(country_data[GDP])
    for i in range(length):
        if i == length - 1:
            continue
        country_data[CPI][length - i - 1] = (
            country_data[CPI][length - i - 1] - country_data[CPI][length - i - 2]
        )
        country_data[IR][length - i - 1] = (
            country_data[IR][length - i - 1] - country_data[IR][length - i - 2]
        )

    # * remove the first element of each indicator
    for indicator in Indicator:
        country_data[indicator] = country_data[indicator][1:]
    return country_data


def construct_starting_markov_chain(
    country_data: dict[Indicator, Matrix[Literal["N"], Float]]
) -> tuple[MarkovChain, MarkovChain]:
    """
    Construct the starting markov chain from the country data
    """
    length = len(country_data[GDP])
    hidden_states = np.zeros(4)
    known_var_states = np.zeros(2)
    hidden_transitions = np.zeros((4, 4))
    known_var_transitions = np.zeros((4, 2))

    # * count the number of each state and transition
    for i in range(length):
        hidden_state: HiddenState = HiddenState.get_state(
            country_data[IR][i], country_data[CPI][i]
        )
        known_var_state: KnownVariables = KnownVariables.get_variable(
            country_data[GDP][i]
        )

        hidden_states[hidden_state.value] += 1
        known_var_states[known_var_state.value] += 1

        if i == length - 1:
            continue
        next_hidden_state: HiddenState = HiddenState.get_state(
            country_data[IR][i + 1], country_data[CPI][i + 1]
        )
        next_known_var_state: KnownVariables = KnownVariables.get_variable(
            country_data[GDP][i + 1]
        )

        hidden_transitions[hidden_state.value][next_hidden_state.value] += 1
        known_var_transitions[hidden_state.value][next_known_var_state.value] += 1

    # * normalize the probabilities
    hidden_states = hidden_states / np.sum(hidden_states)
    known_var_states = known_var_states / np.sum(known_var_states)
    for i in range(4):
        hidden_transitions[i] = hidden_transitions[i] / np.sum(hidden_transitions[i])
        known_var_transitions[i] = known_var_transitions[i] / np.sum(
            known_var_transitions[i]
        )

    # * create the markov chains
    hidden_markov_chain: MarkovChain = MarkovChain(
        states=hidden_states, transitions=hidden_transitions, initial_state=0
    )
    known_var_markov_chain: MarkovChain = MarkovChain(
        states=known_var_states, transitions=known_var_transitions, initial_state=0
    )

    return hidden_markov_chain, known_var_markov_chain


def _forward(
    start_probabilities: Matrix[Literal["N"], Float],
    transition_matrix: Matrix[Literal["N N"], Float],
    emission_matrix: Matrix[Literal["M M"], Float],
    observations: list[int],
) -> Matrix[Literal["N"], Float]:
    """
    Forward algorithm for Hidden Markov Models.

    Parameters:
        start_probabilities: The initial probabilities of the hidden states.
        transition_matrix: The transition probabilities between the hidden states.
        emission_matrix: The emission probabilities of the observations given the hidden states.
        observations: The observations.

    Returns:
        The forward probabilities of the hidden states given the observations.
    """
    num_hidden_states = len(start_probabilities)
    num_observations = len(observations)
    alpha: Matrix[Literal["N N"], Float] = np.zeros(
        (num_hidden_states, num_observations)
    )

    # Initialization
    for i in range(num_hidden_states):
        alpha[i][0] = start_probabilities[i] * emission_matrix[i][observations[0]]

    # Recursion
    for t in range(1, num_observations):
        for i in range(num_hidden_states):
            alpha[i][t] = np.sum(
                [
                    alpha[j][t - 1]
                    * transition_matrix[j][i]
                    * emission_matrix[i][observations[t]]
                    for j in range(num_hidden_states)
                ]
            )

    return alpha


def _backward(
    transition_matrix: Matrix[Literal["N N"], Float],
    emission_matrix: Matrix[Literal["M M"], Float],
    observations: list[int],
) -> Matrix[Literal["N"], Float]:
    """
    Backward algorithm for Hidden Markov Models.

    Parameters:
        transition_matrix: The transition probabilities between the hidden states.
        emission_matrix: The emission probabilities of the observations given the hidden states.
        observations: The observations.

    Returns:
        The backward probabilities of the hidden states given the observations.
    """
    num_hidden_states = len(transition_matrix)
    num_observations = len(observations)
    beta: Matrix[Literal["N N"], Float] = np.zeros(
        (num_hidden_states, num_observations)
    )

    # Initialization
    for i in range(num_hidden_states):
        beta[i][-1] = 1

    # Recursion
    for t in range(num_observations - 2, -1, -1):
        for i in range(num_hidden_states):
            beta[i][t] = np.sum(
                [
                    beta[j][t + 1]
                    * transition_matrix[i][j]
                    * emission_matrix[j][observations[t + 1]]
                    for j in range(num_hidden_states)
                ]
            )

    return beta


def _compute_temporary_variables(
    alpha: Matrix[Literal["N N"], Float],
    beta: Matrix[Literal["N N"], Float],
    transition_matrix: Matrix[Literal["N N"], Float],
    emission_matrix: Matrix[Literal["M M"], Float],
    observations: list[int],
) -> tuple[Matrix[Literal["N N"], Float], Matrix[Literal["N N N"], Float]]:
    """
    Compute the gamma and xi probabilities for the Hidden Markov Model.

    Parameters:
        alpha: The forward probabilities of the hidden states given the observations.
        beta: The backward probabilities of the hidden states given the observations.
        transition_matrix: The transition probabilities between the hidden states.
        emission_matrix: The emission probabilities of the observations given the hidden states.
        observations: The observations.

    Returns:
        The gamma and xi probabilities of the hidden states given the observations.
    """

    num_hidden_states = len(alpha)
    num_observations = len(alpha[0])
    gamma: Matrix[Literal["N N"], Float] = np.zeros(
        (num_hidden_states, num_observations)
    )

    for t in range(num_observations):
        denominator = np.sum(
            [alpha[j][t] * beta[j][t] for j in range(num_hidden_states)]
        )
        for i in range(num_hidden_states):
            numerator = alpha[i][t] * beta[i][t]
            gamma[i][t] = numerator / denominator if denominator != 0 else 0

    xi = np.zeros((num_hidden_states, num_hidden_states, num_observations - 1))
    for t in range(num_observations - 1):
        denominator = np.sum(
            [
                alpha[k][t]
                * transition_matrix[k][w]
                * beta[w][t + 1]
                * emission_matrix[w][observations[t + 1]]
                for w in range(num_hidden_states)
                for k in range(num_hidden_states)
            ]
        )
        for i in range(num_hidden_states):
            for j in range(num_hidden_states):
                numerator = (
                    alpha[i][t]
                    * transition_matrix[i][j]
                    * beta[j][t + 1]
                    * emission_matrix[j][observations[t + 1]]
                )
                xi[i][j][t] = numerator / denominator if denominator != 0 else 0

    return gamma, xi
