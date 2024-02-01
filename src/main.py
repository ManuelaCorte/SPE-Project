import warnings

import numpy as np

from src.data import clean_inflation_dataset, convert_to_matrix, get_time_periods_colums
from src.models import MarkovChain
from src.structs import Country, Indicator, TimePeriod
from src.visualization import autocorrelation_plots, plot_time_series

warnings.simplefilter(action="ignore", category=DeprecationWarning)

if __name__ == "__main__":
    df = clean_inflation_dataset(Country.ITALY, TimePeriod.YEAR, save_intermediate=True)
    # df = pd.read_csv("data/cleaned/italy/year_data.csv")
    X = np.array(get_time_periods_colums(df.columns))
    Y = convert_to_matrix(df, Indicator.CPI)[1, :]

    plot_time_series(X, Y, "IR", 5)
    autocorrelation_plots(X, Y, 10, "IR")

    states = np.array([0.5, 0.5])
    transistions = np.array([[0.1, 0.9], [0.3, 0.7]])
    mc = MarkovChain(states, transistions, 0)
    mc.to_image("markov_chain")
