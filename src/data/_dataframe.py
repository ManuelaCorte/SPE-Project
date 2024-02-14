import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.structs import Indicator
from src.utils import Float, Matrix


def get_indicators_columns(columns: Any | list[str]) -> list[str]:
    """
    Get the columns containing the economic indicators

    Parameters:
        columns: The columns of the dataframe

    Returns:
        The columns containing the economic indicators
    """
    if isinstance(columns, pd.Index):
        columns = columns.values.tolist()

    indicators: list[str] = []
    for column in columns:
        year: str = column[:4]
        if not year.isnumeric():
            indicators.append(column)

    return indicators


def get_time_periods_colums(columns: Any | list[str]) -> list[str]:
    """
    Get the columns containing the time periods. If those columns represent years
    then they are simply the years while for quarters (months) they are equal to
    YYYYQ1 (YYYYMM).

    Parameters:
        columns: The columns of the dataframe

    Returns:
        The columns containing the time periods
    """
    if isinstance(columns, pd.Index):
        columns = columns.values.tolist()

    time_periods: list[str] = []
    for column in columns:
        year = column[:4]
        if year.isnumeric():
            time_periods.append(column)

    return time_periods


def convert_to_matrix(
    df: pd.DataFrame, indicator: Indicator
) -> Matrix[Literal["M N"], Float]:
    """
    Convert the dataframe to a matrix containing the values of the given indicator

    Parameters:
        df: The dataframe
        indicator: The indicator

    Returns:
        The matrix containing the values of the given indicator
    """
    # Get row corresponding to the given indicator
    dataframe = df[df["Indicator Name"].str.contains(indicator.value)]
    # Get only the rows corresponding to the time periods
    colums: list[str] = get_time_periods_colums(dataframe.columns)
    dataframe = dataframe[colums]

    if len(dataframe > 1):
        warnings.warn(f"Multiple rows refer to indicator {indicator.value}")
    return dataframe.to_numpy(dtype=np.float32)
