from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from src.structs import Country, Indicator
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
    df: pd.DataFrame, indicator: Indicator, country: Optional[Country] = None
) -> Matrix[Literal["N"], Float]:
    """
    Convert the dataframe to a matrix containing the values of the given indicator

    Parameters:
        df: The dataframe
        indicator: The indicator to be extracted
        country: The country to be extracted

    Returns:
        The matrix containing the values of the given indicator
    """
    # Get row corresponding to the given indicator
    dataframe = df[df["Indicator Name"] == indicator.value]

    if country is not None:
        # Get only the row corresponding to the country
        dataframe = dataframe[dataframe["Country Code"] == country.value]

    return dataframe["Value"].to_numpy()


def convert_to_structured_matrix(
    df: pd.DataFrame, indicator: Indicator, country: Optional[Country] = None
) -> Matrix[Literal["N"], Float]:
    """
    Convert the dataframe to a matrix containing the values of the given indicator. The
    matrix is structured and contains the date associated as well: each row is a tuple
    (value, date).

    Parameters:
        df: The dataframe
        indicator: The indicator
        country: The country to be extracted. If None, all the countries are considered

    Returns:
        The matrix containing the values of the given indicator
    """
    # Get row corresponding to the given indicator
    dataframe = df[df["Indicator Name"] == indicator.value]

    if country is not None:
        # Get only the row corresponding to the country
        dataframe = dataframe[dataframe["Country Code"] == country.value]

    data = dataframe[["Value", "Date"]].to_records(index=False)

    return np.array(data, dtype=[("Value", "f8"), ("Date", "O")])
