import numpy as np
import pandas as pd

from src.structs import Indicator
from src.utils import convert_to_matrix, get_time_periods_colums
from src.visualization import autocorrelation_plots, plot_time_series

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned/italy/year_data.csv")
    X = np.array(get_time_periods_colums(df.columns))
    Y = convert_to_matrix(df, Indicator.GDP)[2, :]

    plot_time_series(X, Y, "GDP", 5)
    autocorrelation_plots(X, Y, 10, "GDP")
