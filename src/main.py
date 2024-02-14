import warnings
from pprint import pprint

import pandas as pd
from matplotlib import pyplot as plt

from src.data import convert_to_matrix, remove_empty_values
from src.statistics import correlation, stationarity
from src.structs import Indicator, PlotOptions

warnings.simplefilter(action="ignore", category=DeprecationWarning)

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned/united_states/filtered_by_month.csv")
    df = remove_empty_values(df, "out.csv")
    IR = convert_to_matrix(df, Indicator.IR)
    CPI = convert_to_matrix(df, Indicator.CPIDX)

    adf, kpss = stationarity(CPI.squeeze())

    pprint(
        correlation(
            [IR.squeeze(), CPI.squeeze()],
            PlotOptions("correlation", "Correlation", "IR", "CPI", [], False),
        )
    )
    plt.show()
