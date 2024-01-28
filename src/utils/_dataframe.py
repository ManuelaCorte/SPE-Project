import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.structs import Indicator


def get_indicators_columns(colums: pd.Index | list[str]) -> list[str]:
    if isinstance(colums, pd.Index):
        colums = colums.values.tolist()

    indicators: list[str] = []
    for column in colums:
        year = column[:4]
        if not year.isnumeric():
            indicators.append(column)

    return indicators


def get_time_periods_colums(columns: pd.Index | list[str]) -> list[str]:
    if isinstance(columns, pd.Index):
        columns = columns.values.tolist()

    time_periods: list[str] = []
    for column in columns:
        year = column[:4]
        if year.isnumeric():
            time_periods.append(column)

    return time_periods


def convert_to_matrix(df: pd.DataFrame, indicator: Indicator) -> NDArray[np.float32]:
    # Get row corresponding to the given indicator
    dataframe = df[df["Indicator Name"].str.contains(indicator.value)]
    # Get only the rows corresponding to the time periods
    colums: list[str] = get_time_periods_colums(dataframe.columns)
    dataframe = dataframe[colums]

    dataframe = dataframe.dropna(axis=0, how="all")

    return dataframe.to_numpy(dtype=np.float32)
