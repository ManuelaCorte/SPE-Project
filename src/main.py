import warnings

import numpy as np
from matplotlib import pyplot as plt

from src.data import (
    clean_gdp_dataset,
    clean_inflation_dataset,
    convert_to_matrix,
    get_time_periods_colums,
)
from src.models import DynamicTimeWarping
from src.structs import Country, Indicator, PlotOptions, TimePeriod

warnings.simplefilter(action="ignore", category=DeprecationWarning)

if __name__ == "__main__":
    df = clean_inflation_dataset(Country.ITALY, TimePeriod.YEAR, save_intermediate=True)
    years = np.array(get_time_periods_colums(df.columns))
    IR = convert_to_matrix(df, Indicator.CPI)[1, :]

    df = clean_gdp_dataset(Country.ITALY, TimePeriod.YEAR, save_intermediate=True)
    GDP = convert_to_matrix(df, Indicator.GDP)[1, :]

    dtw = DynamicTimeWarping(IR, GDP)
    distance = dtw.compute_distance()
    dtw.plot(PlotOptions("dtw", "DTW", "IR", "GDP", [], True))
    plt.show()
