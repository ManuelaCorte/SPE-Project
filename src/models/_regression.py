# Regression model predicting gdp based on cpi and ir

# Average data over states and then regression??

# ARMAV model where gdp_t = (cpi_t_it, ir_t_it, cpi_t_jp, ir_t_jp, ...) + error_t ???

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.graphics.gofplots import qqplot

from src.data import serialize_country_data
from src.statistics import differencing
from src.structs import Country, Indicator
from src.utils import PlotOptions, plot_time_series


def regression_model(df: pd.DataFrame):
    features, _ = serialize_country_data(df, Country.UNITED_STATES, pct=True)
    gdp = differencing(features["GDP"].to_numpy(), 1)  # type: ignore
    cpi = differencing(features["CPI"].to_numpy(), 1)  # type: ignore
    ir = differencing(features["IR"].to_numpy(), 1)  # type: ignore
    # Fit the model IR, CPI -> GDP
    model = OLS(gdp, add_constant(np.column_stack((cpi, ir)))).fit()

    return model


if __name__ == "__main__":
    from src.structs import Country

    df = pd.read_csv("data/cleaned/dataset.csv")

    features, years = serialize_country_data(df, Country.UNITED_STATES, pct=True)
    y = features[Indicator.IR]
    # y = differencing(y.to_numpy(), 1)

    plot_time_series(years, y, 11, PlotOptions("", "CPI", "Date", "val", ["CPI", "Average"], False))  # type: ignore

    # model = regression_model(df)
    # print(model.summary())

    # # plot the residuals
    # fig, ax = plt.subplots(1,2, figsize=(20,10))
    # ax[0].plot(model.resid, 'o')
    # ax[0].axhline(y=0, color='r')
    # ax[0].set_xticks([])
    # ax[0].set_xlabel("Residuals")

    # # qq plot
    # qqplot(model.resid, line='s', ax=ax[1])
    plt.show()
