import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from enum import Enum
from tqdm import tqdm

from src.data import clean_dataset, convert_to_structured_matrix
from src.structs import Country, Indicator
from src.models import MarkovChain

GDP, IR, CPI = Indicator

class HiddenState(Enum):
    '''
        The hidden state of the markov chain. Each state tells if the Interest Rate and the CPI are increasing or decreasing
    '''
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
    '''
        The known variables of the markov chain. Each state tells if the GDP is increasing or decreasing
    '''
    I_GDP = 0
    D_GDP = 1

    @staticmethod
    def get_all_variables() -> list[int]:
        return [var.value for var in KnownVariables]
    
    @staticmethod
    def get_variable(gdp: float):
        return KnownVariables.I_GDP if gdp > 0 else KnownVariables.D_GDP

def serialize_country_data(country: Country):
    '''
    take the country data and transform each indicator in a simple series of +1 / -1
    '''
    country_indicators: dict[Indicator, dict[str, list[float]]] = {}
    for indicator in Indicator:
      matrix = convert_to_structured_matrix(df, indicator, country)
      values = [m[0] for m in matrix]
      years: list[float] = []
      for m in matrix:
          year_s = m[1].split("-")
          year_n = int(year_s[0]) * 100 + int(year_s[1])
          years.append(year_n)
      country_indicators[indicator] = {'years': years, 'values': values}

    def getlist(indicator: Indicator):
        return country_indicators[indicator]['years']
    
    # * make sure that all series start from the same date
    while getlist(CPI)[0] != getlist(GDP)[0] or getlist(CPI)[0] != getlist(IR)[0]:
        if getlist(CPI)[0] < getlist(GDP)[0] and getlist(CPI)[0] < getlist(IR)[0]:
            getlist(CPI).pop(0)
            country_indicators[CPI]['values'].pop(0)
        elif getlist(IR)[0] < getlist(GDP)[0]:
            getlist(IR).pop(0)
            country_indicators[IR]['values'].pop(0)
        else: 
            getlist(GDP).pop(0)
            country_indicators[GDP]['values'].pop(0)

    # * make sure that all series end to the same date
    while getlist(CPI)[-1] != getlist(GDP)[-1] or getlist(CPI)[-1] != getlist(IR)[-1]:
        if getlist(CPI)[-1] > getlist(GDP)[-1] and getlist(CPI)[-1] > getlist(IR)[-1]:
            getlist(CPI).pop()
            country_indicators[CPI]['values'].pop()
        elif getlist(IR)[-1] > getlist(GDP)[-1]:
            getlist(IR).pop()
            country_indicators[IR]['values'].pop()
        else: 
            getlist(GDP).pop()
            country_indicators[GDP]['values'].pop()
      
    # * keep only the values and check we have the same number of points for each indicator
    length = len(country_indicators[GDP]['years'])
    country_data: dict[Indicator, list[float]] = {}
    for indicator in Indicator:
        country_data[indicator] = country_indicators[indicator]['values']
        if len(country_data[indicator]) != length:
            raise Exception(f"{country} has different lengths of indicator even if same start and end date")
    
    # * take only the difference between a member and its next
    for i in range(length):
        if i == length - 1: continue
        country_data[CPI][length - i - 1] = country_data[CPI][length - i - 1] - country_data[CPI][length - i - 2]
        country_data[IR][length - i - 1] = country_data[IR][length - i - 1] - country_data[IR][length - i - 2]

    # * since we are taking the difference, we remove the first element
    for indicator in Indicator:
        country_data[indicator].pop(0)

    # * keep only if the difference is positive or negative
    for i in range(length - 1):
        for indicator in Indicator:
            country_data[indicator][i] = 1 if country_data[indicator][i] >= 0 else -1
    return country_data
    
def construct_starting_markov_chain(country_data: dict[Indicator, list[float]]):
    '''
    Construct the starting markov chain from the country data
    '''
    length = len(country_data[GDP])
    hidden_states = np.zeros(4)
    known_var_states = np.zeros(2)
    hidden_transitions = np.zeros((4, 4))
    known_var_transitions = np.zeros((4, 2))

    # * count the number of each state and transition
    for i in range(length):
        hidden_state: HiddenState = HiddenState.get_state(country_data[IR][i], country_data[CPI][i])
        known_var_state: KnownVariables = KnownVariables.get_variable(country_data[GDP][i])

        hidden_states[hidden_state.value] += 1
        known_var_states[known_var_state.value] += 1

        if i == length - 1: continue
        next_hidden_state: HiddenState = HiddenState.get_state(country_data[IR][i + 1], country_data[CPI][i + 1])
        next_known_var_state: KnownVariables = KnownVariables.get_variable(country_data[GDP][i + 1])

        hidden_transitions[hidden_state.value][next_hidden_state.value] += 1
        known_var_transitions[hidden_state.value][next_known_var_state.value] += 1

    # * normalize the probabilities
    hidden_states = hidden_states / np.sum(hidden_states)
    known_var_states = known_var_states / np.sum(known_var_states)
    for i in range(4):
        hidden_transitions[i] = hidden_transitions[i] / np.sum(hidden_transitions[i])
        known_var_transitions[i] = known_var_transitions[i] / np.sum(known_var_transitions[i])

    # * create the markov chains
    hidden_markov_chain: MarkovChain = MarkovChain(
        states=hidden_states, 
        transitions=hidden_transitions, 
        initial_state=0
    )
    known_var_markov_chain: MarkovChain = MarkovChain(
        states=known_var_states, 
        transitions=known_var_transitions, 
        initial_state=0
    )

    return hidden_markov_chain, known_var_markov_chain

if __name__ == "__main__":
    if os.path.exists("data/cleaned/dataset.csv"):
        df = pd.read_csv("data/cleaned/dataset.csv")
    else:
        df = clean_dataset(save_intermediate=True)

    countries_data: dict[Country, dict[Indicator, list[float]]] = {}

    # countries = Country
    countries = [Country.ITALY]
    for country in countries:
      countries_data[country] = serialize_country_data(country)
    # print(countries_data[Country.ITALY])

    starting_country_data = countries_data[Country.ITALY]
    hidden_markov_chain, known_var_markov_chain = construct_starting_markov_chain(starting_country_data)
    print(hidden_markov_chain)
    print(known_var_markov_chain)
    # hidden_markov_chain.to_image_with_known_var("hidden_markov_chain", known_var_markov_chain)

    # * Baum-Welch algorithm

    epochs = 1
    for _ in tqdm(range(epochs), desc='training', unit='epoch'):
        a = hidden_markov_chain.transitions
        b = known_var_markov_chain.transitions
        pi = hidden_markov_chain.states

        gammas: list[npt.NDArray[np.generic]] = []
        xis: list[npt.NDArray[np.generic]] = []
        Ys: list[list[float]] = []
        Ts: list[int] = []
        R = len(countries)

        for country in countries:
            country_data = countries_data[country]
            Y = country_data[GDP]
            Ys.append(Y)

            T = len(Y)
            Ts.append(T)

            alpha = np.zeros((4, T))
            for i in range(4):
                for t in range(T):
                    y = KnownVariables.get_variable(Y[t]).value
                    if t == 0:
                        alpha[i][t] = pi[i] * b[i][y]
                    else:
                        alpha[i][t] = b[i][y] * np.sum([alpha[j][t - 1] * a[j][i] for j in range(4)])

            beta = np.zeros((4, T))
            for i in range(4):
                for t in range(T - 1, -1, -1):
                    if t == T - 1:
                        beta[i][t] = 1
                    else:
                        y = KnownVariables.get_variable(Y[t + 1]).value
                        beta[i][t] = np.sum([a[i][j] * b[j][y] * beta[j][t + 1] for j in range(4)])

            gamma = np.zeros((4, T))
            for t in range(T):
                for i in range(4):
                    num = alpha[i][t] * beta[i][t]
                    den = np.sum([alpha[j][t] * beta[j][t] for j in range(4)])
                    gamma[i][t] = num / den

            xi = np.zeros((4, 4, T - 1))
            for t in range(T - 1):
                for i in range(4):
                    for j in range(4):
                        y = KnownVariables.get_variable(Y[t + 1]).value
                        num = alpha[i][t] * a[i][j] * beta[j][t + 1] * b[j][y]
                        den = np.sum([alpha[k][t] * a[k][w] * beta[k][t+1] * b[w][y] for w in range(4) for k in range(4)])
                        xi[i][j][t] = num / den

            gammas.append(gamma)
            xis.append(xi)

        for i in range(4):
            pi[i] = np.sum([gammas[r][i][0] for r in range(R)]) / R
            for j in range(4):
                num = np.sum([xis[r][i][j][t] for r in range(R) for t in range(Ts[r] - 1)])
                den = np.sum([gammas[r][i][t] for r in range(R) for t in range(Ts[r] - 1)])
                a[i][j] = num / den
            for j in range(2):
                num = np.sum([gammas[r][i][t] for r in range(R) for t in range(Ts[r]) if KnownVariables.get_variable(Ys[r][t]).value == j])
                den = np.sum([gammas[r][i][t] for r in range(R) for t in range(Ts[r])])
                b[i][j] = num / den

        hidden_markov_chain.transitions = a
        known_var_markov_chain.transitions = b
        hidden_markov_chain.states = pi

    print(hidden_markov_chain)
    print(f'hidden states sum: {np.sum(hidden_markov_chain.states):.2f}')
    for i in range(4):
        print(f'hidden transitions {i}: {np.sum(hidden_markov_chain.transitions[i]):.2f}')
    print(known_var_markov_chain)
    for i in range(4):
        print(f'known transitions {i}: {np.sum(known_var_markov_chain.transitions[i]):.2f}')
    # hidden_markov_chain.to_image_with_known_var("hidden_markov_chain_2", known_var_markov_chain)


      
